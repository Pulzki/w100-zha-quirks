"""Quirk for Aqara W100 Climate Sensor with 3 buttons."""
import asyncio
from enum import IntEnum
import logging
import random
import struct
import time
from typing import Any, List, Optional, Union

from zigpy.profiles import zha
from zigpy.quirks.v2 import CustomDeviceV2, QuirkBuilder
import zigpy.types as t
from zigpy.zcl import Cluster, foundation
from zigpy.zcl.clusters.general import (
    Basic,
    MultistateInput,
    OnOff,
    PowerConfiguration,
)
from zigpy.zcl.clusters.hvac import Fan, Thermostat
from zigpy.zcl.clusters.measurement import TemperatureMeasurement

from zhaquirks import CustomCluster
from zhaquirks.const import (
    COMMAND,
    COMMAND_DOUBLE,
    COMMAND_HOLD,
    COMMAND_RELEASE,
    COMMAND_SINGLE,
    DOUBLE_PRESS,
    ENDPOINT_ID,
    LONG_PRESS,
    LONG_RELEASE,
    SHORT_PRESS,
    VALUE,
    ZHA_SEND_EVENT,
)
from zhaquirks.xiaomi import XiaomiAqaraE1Cluster, XiaomiPowerConfigurationPercent

_LOGGER = logging.getLogger(__name__)

# Attribute where button press values are reported
STATUS_TYPE_ATTR = 0x0055
# Custom attribute for W100 control
W100_ATTR = 0xFFF2

# Buttons
PLUS_BUTTON = "plus"
CENTER_BUTTON = "center"
MINUS_BUTTON = "minus"

# Map reported values to action names
PRESS_TYPES = {
    0: COMMAND_HOLD,
    1: COMMAND_SINGLE,
    2: COMMAND_DOUBLE,
    255: COMMAND_RELEASE,
}

# Optional label per endpoint
BUTTON_NAMES = {
    1: PLUS_BUTTON,
    2: CENTER_BUTTON,
    3: MINUS_BUTTON,
}


class MultistateInputCluster(CustomCluster, MultistateInput):
    """MultistateInput cluster that emits zha_event with button and press type."""

    def __init__(self, *args, **kwargs):
        """Initialize the cluster."""
        self._current_state = None
        super().__init__(*args, **kwargs)

    def _update_attribute(self, attrid, value):
        super()._update_attribute(attrid, value)

        if attrid == STATUS_TYPE_ATTR:
            command = PRESS_TYPES.get(value, COMMAND_SINGLE)
            event_args = {
                COMMAND: command,
                VALUE: value,
                ENDPOINT_ID: self.endpoint.endpoint_id,
                "args": BUTTON_NAMES.get(self.endpoint.endpoint_id, "unknown"),
            }
            self.listener_event(ZHA_SEND_EVENT, command, event_args)


class W100TemperatureMeasurement(CustomCluster, TemperatureMeasurement):
    """Temperature cluster that updates thermostat local temperature."""

    def _update_attribute(self, attrid, value):
        super()._update_attribute(attrid, value)
        if attrid == 0x0000:  # measured_value
            # Update thermostat local temperature
            thermostat = self.endpoint.in_clusters.get(Thermostat.cluster_id)
            if thermostat:
                thermostat.update_attribute(0x0000, value)
                if hasattr(thermostat, "recalculate_running_state"):
                    thermostat.recalculate_running_state()


class W100ExternalSensorMode(IntEnum):
    """W100 External Sensor Mode."""
    internal = 0
    external = 2


class W100ManuSpecificCluster(XiaomiAqaraE1Cluster):
    """Aqara W100 custom cluster."""

    manufacturer_id_override = 0x115F

    class AttributeDefs(XiaomiAqaraE1Cluster.AttributeDefs):
        """Attribute definitions."""

        period = foundation.ZCLAttributeDef(id=0x0162, type=t.uint32_t, is_manufacturer_specific=True)
        temp_period = foundation.ZCLAttributeDef(id=0x0163, type=t.uint32_t, is_manufacturer_specific=True)
        temp_threshold = foundation.ZCLAttributeDef(id=0x0164, type=t.uint16_t, is_manufacturer_specific=True)
        temp_report_mode = foundation.ZCLAttributeDef(id=0x0165, type=t.uint8_t, is_manufacturer_specific=True)
        low_temperature = foundation.ZCLAttributeDef(id=0x0166, type=t.int16s, is_manufacturer_specific=True)
        high_temperature = foundation.ZCLAttributeDef(id=0x0167, type=t.int16s, is_manufacturer_specific=True)
        humi_period = foundation.ZCLAttributeDef(id=0x016A, type=t.uint32_t, is_manufacturer_specific=True)
        humi_threshold = foundation.ZCLAttributeDef(id=0x016B, type=t.uint16_t, is_manufacturer_specific=True)
        humi_report_mode = foundation.ZCLAttributeDef(id=0x016C, type=t.uint8_t, is_manufacturer_specific=True)
        low_humidity = foundation.ZCLAttributeDef(id=0x016D, type=t.int16s, is_manufacturer_specific=True)
        high_humidity = foundation.ZCLAttributeDef(id=0x016E, type=t.int16s, is_manufacturer_specific=True)
        sampling = foundation.ZCLAttributeDef(id=0x0170, type=t.uint8_t, is_manufacturer_specific=True)
        sensor = foundation.ZCLAttributeDef(id=0x0172, type=t.uint8_t, is_manufacturer_specific=True)
        auto_hide_middle_line = foundation.ZCLAttributeDef(id=0x0173, type=t.Bool, is_manufacturer_specific=True)
        external_temperature = foundation.ZCLAttributeDef(id=0x0174, type=t.Single, is_manufacturer_specific=True)
        external_humidity = foundation.ZCLAttributeDef(id=0x0175, type=t.Single, is_manufacturer_specific=True)
        w100_control = foundation.ZCLAttributeDef(id=0xFFF2, type=t.LVBytes, is_manufacturer_specific=True)
        thermostat_mode_switch = foundation.ZCLAttributeDef(id=0xFFF3, type=t.Bool, is_manufacturer_specific=True)
        unknown_df = foundation.ZCLAttributeDef(id=0x00DF, type=t.LVBytes, is_manufacturer_specific=True)
        diagnostics = foundation.ZCLAttributeDef(id=0x00F7, type=t.LVBytes, is_manufacturer_specific=True)

    _FICTIVE_SENSOR_IEEE = bytes.fromhex("00158d00019d1b98")

    # Default values for external sensor measurements
    _DEFAULT_EXTERNAL_TEMPERATURE = 20.0
    _DEFAULT_EXTERNAL_HUMIDITY = 50.0

    def __init__(self, *args, **kwargs):
        """Initialize the cluster with cached external sensor values."""
        super().__init__(*args, **kwargs)
        self._cached_external_temperature = self._DEFAULT_EXTERNAL_TEMPERATURE
        self._cached_external_humidity = self._DEFAULT_EXTERNAL_HUMIDITY
        # Initialize the attribute cache with default values
        self._update_attribute(self.AttributeDefs.external_temperature.id, self._cached_external_temperature)
        self._update_attribute(self.AttributeDefs.external_humidity.id, self._cached_external_humidity)

    @staticmethod
    def _lumi_header(counter: int, length: int, action: int) -> bytes:
        header = [0xAA, 0x71, length + 3, 0x44, counter & 0xFF]
        integrity = (512 - sum(header)) & 0xFF
        return bytes(header + [integrity, action & 0xFF, 0x41, length & 0xFF])

    def _normalize_sensor_mode(self, value: Any) -> int:
        try:
            v = int(value)
        except (TypeError, ValueError):
            return int(W100ExternalSensorMode.internal)
        if v in (2, 3):
            return int(W100ExternalSensorMode.external)
        return int(W100ExternalSensorMode.internal)

    def _device_ieee_bytes(self) -> bytes:
        # Keep consistent with how this quirk already builds W100 frames.
        return self.endpoint.device.ieee.serialize()

    async def _write_w100_control_frame(self, payload: bytes) -> None:
        await super().write_attributes({W100_ATTR: payload}, manufacturer=0x115F)

    async def _set_external_sensor_mode(self, mode: Any) -> None:
        normalized = self._normalize_sensor_mode(mode)

        device = self._device_ieee_bytes()
        timestamp = struct.pack(">I", int(time.time()))

        if normalized == int(W100ExternalSensorMode.external):
            params1 = bytes(
                list(timestamp)
                + [0x15]
                + list(device)
                + list(self._FICTIVE_SENSOR_IEEE)
                + [
                    0x00,
                    0x02,
                    0x00,
                    0x55,
                    0x15,
                    0x0A,
                    0x01,
                    0x00,
                    0x00,
                    0x01,
                    0x06,
                    0xE6,
                    0xB9,
                    0xBF,
                    0xE5,
                    0xBA,
                    0xA6,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x01,
                    0x02,
                    0x08,
                    0x65,
                ]
            )
            params2 = bytes(
                list(timestamp)
                + [0x14]
                + list(device)
                + list(self._FICTIVE_SENSOR_IEEE)
                + [
                    0x00,
                    0x01,
                    0x00,
                    0x55,
                    0x15,
                    0x0A,
                    0x01,
                    0x00,
                    0x00,
                    0x01,
                    0x06,
                    0xE6,
                    0xB8,
                    0xA9,
                    0xE5,
                    0xBA,
                    0xA6,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x01,
                    0x02,
                    0x07,
                    0x63,
                ]
            )

            val1 = self._lumi_header(0x12, len(params1), 0x02) + params1
            val2 = self._lumi_header(0x13, len(params2), 0x02) + params2

            await self._write_w100_control_frame(val1)
            await self._write_w100_control_frame(val2)
            
            # Small delay to ensure mode switch is processed
            await asyncio.sleep(0.3)
            await self._send_external_measurement("temperature", self._cached_external_temperature)
            await asyncio.sleep(0.1)
            await self._send_external_measurement("humidity", self._cached_external_humidity)
        else:
            params1 = timestamp + bytes([0x15]) + device + (b"\x00" * 12)
            params2 = timestamp + bytes([0x14]) + device + (b"\x00" * 12)

            val1 = self._lumi_header(0x12, len(params1), 0x04) + params1
            val2 = self._lumi_header(0x13, len(params2), 0x04) + params2

            await self._write_w100_control_frame(val1)
            await self._write_w100_control_frame(val2)
        try:
            await super().read_attributes([self.AttributeDefs.sensor.id], manufacturer=0x115F)
        except Exception:
            # Ignore read failures; UI will still reflect the last value written.
            pass

        self._update_attribute(self.AttributeDefs.sensor.id, normalized)

    async def _send_external_measurement(self, kind: str, value: Any) -> None:
        sensor_mode = self.get(self.AttributeDefs.sensor.id)
        if sensor_mode is not None and self._normalize_sensor_mode(sensor_mode) != int(W100ExternalSensorMode.external):
            _LOGGER.debug("W100: Ignoring %s update while sensor mode is internal", kind)
            return

        number = float(value)
        measurement = struct.pack(">f", float(round(number * 100)))

        if kind == "temperature":
            params = self._FICTIVE_SENSOR_IEEE + bytes([0x00, 0x01, 0x00, 0x55]) + measurement
        elif kind == "humidity":
            params = self._FICTIVE_SENSOR_IEEE + bytes([0x00, 0x02, 0x00, 0x55]) + measurement
        else:
            raise ValueError(f"Unknown kind: {kind}")

        data = self._lumi_header(0x12, len(params), 0x05) + params
        await self._write_w100_control_frame(data)

    async def write_attributes(self, attributes, manufacturer=None):
        """Intercept write attributes."""
        if "sensor" in attributes or self.AttributeDefs.sensor.id in attributes:
            val = attributes.get("sensor", attributes.get(self.AttributeDefs.sensor.id))
            await self._set_external_sensor_mode(val)
            return [
                [
                    foundation.WriteAttributesStatusRecord(
                        foundation.Status.SUCCESS,
                        attrid=self.AttributeDefs.sensor.id,
                    )
                ]
            ]

        if "external_temperature" in attributes or self.AttributeDefs.external_temperature.id in attributes:
            val = attributes.get(
                "external_temperature", attributes.get(self.AttributeDefs.external_temperature.id)
            )
            # Cache the value for use when switching to external mode
            self._cached_external_temperature = float(val)
            _LOGGER.debug("W100: Cached external temperature: %.1f", self._cached_external_temperature)
            try:
                await self._send_external_measurement("temperature", val)
            finally:
                # Keep last value in cache for the HA UI
                self._update_attribute(self.AttributeDefs.external_temperature.id, float(val))
            return [
                [
                    foundation.WriteAttributesStatusRecord(
                        foundation.Status.SUCCESS,
                        attrid=self.AttributeDefs.external_temperature.id,
                    )
                ]
            ]

        if "external_humidity" in attributes or self.AttributeDefs.external_humidity.id in attributes:
            val = attributes.get("external_humidity", attributes.get(self.AttributeDefs.external_humidity.id))
            # Cache the value for use when switching to external mode
            self._cached_external_humidity = float(val)
            _LOGGER.debug("W100: Cached external humidity: %.1f", self._cached_external_humidity)
            try:
                await self._send_external_measurement("humidity", val)
            finally:
                self._update_attribute(self.AttributeDefs.external_humidity.id, float(val))
            return [
                [
                    foundation.WriteAttributesStatusRecord(
                        foundation.Status.SUCCESS,
                        attrid=self.AttributeDefs.external_humidity.id,
                    )
                ]
            ]

        if "thermostat_mode_switch" in attributes or 0xFFF3 in attributes:
            val = attributes.get("thermostat_mode_switch", attributes.get(0xFFF3))
            mode = "ON" if val else "OFF"
            await self.set_thermostat_mode(mode)
            self._update_attribute(0xFFF3, val)
            
            records = [
                foundation.WriteAttributesStatusRecord(
                    foundation.Status.SUCCESS,
                    attrid=0xFFF3
                )
            ]
            return [records]
            
        return await super().write_attributes(attributes, manufacturer)

    async def read_attributes(
        self, attributes, allow_cache=False, only_cache=False, manufacturer=None
    ):
        """Read attributes with manufacturer code forced."""
        if manufacturer is None:
            manufacturer = 0x115F

        # These are virtual config attributes (used only to trigger writes).
        # Serve them from cache to avoid unsupported attribute reads.
        virtual_ids = {
            self.AttributeDefs.external_temperature.id,
            self.AttributeDefs.external_humidity.id,
        }

        requested_ids: list[int] = []
        passthrough: list[Any] = []
        for attr in attributes:
            if isinstance(attr, str) and attr in self.attributes_by_name:
                attr_id = self.attributes_by_name[attr].id
            elif isinstance(attr, int):
                attr_id = attr
            else:
                passthrough.append(attr)
                continue

            if attr_id in virtual_ids:
                requested_ids.append(attr_id)
            else:
                passthrough.append(attr)

        success: dict[Any, Any] = {}
        failure: dict[Any, Any] = {}

        for attr_id in requested_ids:
            cached = self.get(attr_id)
            if cached is not None:
                success[attr_id] = cached

        if passthrough:
            s, f = await super().read_attributes(
                passthrough, allow_cache, only_cache, manufacturer
            )
            success.update(s)
            failure.update(f)

        return success, failure

    def _update_attribute(self, attrid, value):
        value_hex: Optional[str] = None
        if isinstance(value, (bytes, bytearray, memoryview)):
            value_hex = bytes(value).hex()

        if attrid == self.AttributeDefs.sensor.id:
            value = self._normalize_sensor_mode(value)

        if value_hex is None:
            _LOGGER.debug(
                "W100ManuSpecificCluster: _update_attribute attrid=0x%04X value=%r",
                int(attrid),
                value,
            )
        else:
            _LOGGER.debug(
                "W100ManuSpecificCluster: _update_attribute attrid=0x%04X value=%r hex=%s",
                int(attrid),
                value,
                value_hex,
            )
        
        if attrid == 0x00F7:
            self._update_battery_from_xiaomi_struct(value)

        super()._update_attribute(attrid, value)
        
        if attrid == W100_ATTR:
            self._parse_w100_data(value)
        elif attrid == 0x0173:  # auto_hide_middle_line
            # Value is True(1) or False(0)
            # Switch On = Auto Hide On = 0 (False)
            # Switch Off = Auto Hide Off = 1 (True)
            switch_val = not value
            
            on_off_cluster = self.endpoint.in_clusters.get(OnOff.cluster_id)
            if on_off_cluster:
                on_off_cluster.update_attribute(0x0000, switch_val)

    def _update_battery_from_xiaomi_struct(self, value):
        """Parse Xiaomi struct to extract battery percentage."""
        try:
            if not isinstance(value, (bytes, bytearray, memoryview)):
                return

            # Aqara W100 / TH-S04D exposes battery from Lumi struct key 102.
            # In hex, that's 0x66. Some Xiaomi devices use key 0x05 instead.
            BATTERY_PERCENT_KEY_W100 = 0x66
            BATTERY_PERCENT_KEY_LEGACY = 0x05
            BATTERY_VOLTAGE_KEY = 0x01

            voltage_mv: Optional[int] = None
            battery_percent_66: Optional[int] = None
            battery_percent_05: Optional[int] = None

            data = bytes(value)
            while data not in (b"", b"\x00"):
                tag = data[0]
                svalue, data = foundation.TypeValue.deserialize(data[1:])
                raw = svalue.value

                if tag == BATTERY_VOLTAGE_KEY and isinstance(raw, int):
                    _LOGGER.debug("W100: Parsed battery voltage key 0x%02X: %d mV", tag, raw)
                    voltage_mv = raw
                elif tag == BATTERY_PERCENT_KEY_W100 and isinstance(raw, int):
                    _LOGGER.debug("W100: Parsed battery percent key 0x%02X: %d%%", tag, raw)
                    battery_percent_66 = raw
                elif tag == BATTERY_PERCENT_KEY_LEGACY and isinstance(raw, int):
                    # On some devices this is a uint16 and may be in half-percent steps (0-200).
                    _LOGGER.debug("W100: Parsed battery percent key 0x%02X: %d (raw)", tag, raw)
                    battery_percent_05 = raw

            # Prefer the W100-specific key 0x66.
            battery_percent: Optional[int] = battery_percent_66

            # Fallback to legacy key 0x05 only if 0x66 wasn't present.
            if battery_percent is None and battery_percent_05 is not None:
                normalized = battery_percent_05
                if normalized > 100 and normalized <= 200:
                    normalized = int(round(normalized / 2))
                battery_percent = max(0, min(100, normalized))

            ep1 = self.endpoint.device.endpoints.get(1)
            if not ep1:
                return

            power_cluster = ep1.in_clusters.get(PowerConfiguration.cluster_id)
            if not power_cluster:
                return

            # Use the XiaomiPowerConfiguration* helpers when available so scaling is correct.
            if voltage_mv is not None and hasattr(power_cluster, "battery_reported"):
                power_cluster.battery_reported(voltage_mv)

            if battery_percent is not None and hasattr(power_cluster, "battery_percent_reported"):
                power_cluster.battery_percent_reported(battery_percent)
        except Exception as e:
            _LOGGER.warning("W100: Error parsing 0xF7: %s", e)

    def _parse_w100_data(self, data):
        """Parse the custom W100 data blob."""
        if not data or not isinstance(data, bytes):
            return

        # Look for the end marker [0x08, 0x44]
        end_marker = b"\x08\x44"
        idx = data.find(end_marker)
        if idx == -1:
            return

        if idx + 2 >= len(data):
            # Check for heartbeat/request (0x84 command)
            if len(data) >= 4 and data[3] == 0x84:
                 _LOGGER.debug("W100ManuSpecificCluster: Received heartbeat/request. Sending PMTSD update.")
                 asyncio.create_task(self._send_cached_pmtsd())
            return

        payload_len = data[idx + 2]
        payload_start = idx + 3
        payload_end = payload_start + payload_len

        if payload_end > len(data):
            return

        try:
            payload_ascii = data[payload_start:payload_end].decode("ascii")
        except UnicodeDecodeError:
            _LOGGER.debug("W100ManuSpecificCluster: Ignoring non-ASCII payload")
            return

        _LOGGER.debug("W100 payload: %s", payload_ascii)
        self._process_payload_updates(payload_ascii)

    def _process_payload_updates(self, payload_ascii):
        """Process the parsed ASCII payload."""
        # Parse P_M_T_S_D format
        updates = {}
        for pair in payload_ascii.split("_"):
            if len(pair) < 2:
                continue
            try:
                updates[pair[0]] = float(pair[1:])
            except ValueError:
                continue

        # Update Thermostat on Endpoint 1
        ep1 = self.endpoint.device.endpoints.get(1)
        if not ep1:
            return
            
        thermostat = ep1.out_clusters.get(Thermostat.cluster_id) or ep1.in_clusters.get(Thermostat.cluster_id)

        if thermostat and isinstance(thermostat, W100ThermostatCluster):
            self._update_thermostat_cache(thermostat, updates)
            
            # Update Fan on Endpoint 21
            if "S" in updates:
                self._update_fan_cluster(int(updates["S"]))

            # Update System Mode
            self._update_system_mode(thermostat, updates)

    def _update_thermostat_cache(self, thermostat, updates):
        """Update thermostat cached values."""
        if "P" in updates:
            thermostat._cached_p = int(updates["P"])
        if "M" in updates:
            thermostat._cached_m = int(updates["M"])
        if "T" in updates:
            thermostat._cached_t = updates["T"]
            # Update setpoints
            val = updates["T"] * 100
            thermostat.update_attribute(
                Thermostat.attributes_by_name["occupied_heating_setpoint"].id,
                val,
            )
            thermostat.update_attribute(
                Thermostat.attributes_by_name["occupied_cooling_setpoint"].id,
                val,
            )
        if "S" in updates:
            thermostat._cached_s = int(updates["S"])
        if "D" in updates:
            thermostat._cached_d = updates["D"]
            
        thermostat.recalculate_running_state()

    def _update_fan_cluster(self, s_value):
        """Update fan cluster based on S value."""
        ep21 = self.endpoint.device.endpoints.get(21)
        if ep21:
            fan = ep21.in_clusters.get(Fan.cluster_id)
            if fan:
                mode = Fan.FanMode.Auto
                if s_value == 1:
                    mode = Fan.FanMode.Low
                elif s_value == 2:
                    mode = Fan.FanMode.Medium
                elif s_value == 3:
                    mode = Fan.FanMode.High
                
                _LOGGER.debug("W100: Updating fan mode to %s (S=%s)", mode, s_value)
                fan.update_attribute(0x0000, mode)

    def _update_system_mode(self, thermostat, updates):
        """Update system mode based on P and M values."""
        p = thermostat._cached_p
        m = thermostat._cached_m
        
        # Override with updates if present
        if "P" in updates:
            p = int(updates["P"])
        if "M" in updates:
            m = int(updates["M"])
        
        _LOGGER.debug("W100: Syncing system_mode with P=%s, M=%s", p, m)
            
        if p is not None and m is not None:
            if p == 1:
                mode = Thermostat.SystemMode.Off
            elif m == 0:
                mode = Thermostat.SystemMode.Cool
            elif m == 1:
                mode = Thermostat.SystemMode.Heat
            elif m == 2:
                mode = Thermostat.SystemMode.Auto
            else:
                mode = Thermostat.SystemMode.Off
            
            _LOGGER.debug("W100: Updating system_mode to %s", mode)
            thermostat.update_attribute(
                Thermostat.attributes_by_name["system_mode"].id,
                mode
            )
            thermostat.recalculate_running_state()

    async def _send_cached_pmtsd(self):
        """Send cached PMTSD to device."""
        ep1 = self.endpoint.device.endpoints.get(1)
        if ep1:
            thermostat = ep1.in_clusters.get(Thermostat.cluster_id)
            if thermostat and hasattr(thermostat, "_cached_p"):
                 await self.send_pmtsd(
                    thermostat._cached_p,
                    thermostat._cached_m,
                    thermostat._cached_t,
                    thermostat._cached_s,
                    thermostat._cached_d
                )

    async def send_pmtsd(self, p, m, t_val, s, d):
        """Send PMTSD command."""
        # Format: P{P}_M{M}_T{T}_S{S}_D{D}
        pmtsd_str = f"P{p}_M{m}_T{t_val}_S{s}_D{d}"
        pmtsd_bytes = pmtsd_str.encode("ascii")
        pmtsd_len = len(pmtsd_bytes)

        fixed_header = bytearray([
            0xAA, 0x71, 0x1F, 0x44,
            0x00, 0x00, 0x05, 0x41, 0x1C,
            0x00, 0x00,
            0x54, 0xEF, 0x44, 0x80, 0x71, 0x1A,
            0x08, 0x00, 0x08, 0x44, pmtsd_len,
        ])

        counter = random.randint(0, 255)
        fixed_header[4] = counter

        full_payload = fixed_header + pmtsd_bytes
        
        # Checksum calculation: sum of all bytes & 0xFF
        checksum = sum(full_payload) & 0xFF
        full_payload[5] = checksum

        await self.write_attributes(
            {W100_ATTR: full_payload},
            manufacturer=0x115F,
        )

    async def set_thermostat_mode(self, mode: str):
        """Set thermostat mode (ON/OFF)."""
        device_ieee = self.endpoint.device.ieee
        dev_mac = device_ieee.serialize()
        hub_mac = bytes.fromhex("54ef4480711a")

        if mode == "ON":
            prefix = bytes.fromhex("aa713244")
            message_alea = bytes([random.randint(0, 255), random.randint(0, 255)])
            zigbee_header = bytes.fromhex("02412f6891")
            message_id = bytes([random.randint(0, 255), random.randint(0, 255)])
            control = b"\x18"
            
            payload_macs = dev_mac + b"\x00\x00" + hub_mac
            payload_tail = bytes.fromhex("08000844150a0109e7a9bae8b083e58a9f000000000001012a40")
            
            frame = prefix + message_alea + zigbee_header + message_id + control + payload_macs + payload_tail
        else:
            prefix = bytes.fromhex("aa711c44691c0441196891")
            frame_id = bytes([random.randint(0, 255)])
            seq = bytes([random.randint(0, 255)])
            control = b"\x18"
            
            frame = prefix + frame_id + seq + control + dev_mac
            if len(frame) < 34:
                frame += b"\x00" * (34 - len(frame))

        await self.write_attributes(
            {W100_ATTR: frame},
            manufacturer=0x115F,
        )

    async def apply_custom_configuration(self, *args, **kwargs):
        """Configure the device."""
        _LOGGER.debug("W100: Configuring device via apply_custom_configuration")
        
        # Configure Reporting
        try:
            # Standard Temp
            temp_cluster = self.endpoint.in_clusters.get(TemperatureMeasurement.cluster_id)
            if temp_cluster:
                await temp_cluster.bind()
                await temp_cluster.configure_reporting(0x0000, 10, 3600, 50)
            
            # Custom Temp
            await self.configure_reporting(0x0163, 10, 3600, 100, manufacturer=0x115F)
            
            # Custom Humidity
            await self.configure_reporting(0x016A, 10, 3600, 100, manufacturer=0x115F)
            
            # Power
            power_cluster = self.endpoint.in_clusters.get(PowerConfiguration.cluster_id)
            if power_cluster:
                await power_cluster.bind()
                await power_cluster.configure_reporting(0x0020, 3600, 21600, 1)
        except Exception as e:
            _LOGGER.warning("W100: Failed to configure reporting: %s", e)

        # Initialize Mode
        try:
            await self.bind()
            await self.set_thermostat_mode("OFF")
            await asyncio.sleep(0.5)
            await self.set_thermostat_mode("ON")
        except Exception as e:
            _LOGGER.warning("W100: Failed to set thermostat mode: %s", e)

        # Sync state
        try:
            await self.read_attributes([W100_ATTR], manufacturer=0x115F)
            if temp_cluster:
                await temp_cluster.read_attributes([0x0000])
        except Exception as e:
            _LOGGER.warning("W100: Failed to sync state: %s", e)

class W100BasicCluster(CustomCluster, Basic):
    """Basic cluster that handles ReportAttributes as ReadAttributesResponse."""

    async def read_attributes(
        self, attributes, allow_cache=False, only_cache=False, manufacturer=None
    ):
        """Read attributes."""
        if only_cache:
            return await super().read_attributes(
                attributes, allow_cache, only_cache, manufacturer
            )

        try:
            results = await self.read_attributes_raw(attributes, manufacturer=manufacturer)
        except Exception:
            return await super().read_attributes(
                attributes, allow_cache, only_cache, manufacturer
            )

        success = {}
        failure = {}

        for record in results:
            if isinstance(record, foundation.Attribute):
                success[record.attrid] = record.value.value
            else:
                if record.status == foundation.Status.SUCCESS:
                    success[record.attrid] = record.value.value
                else:
                    failure[record.attrid] = record.status

        return success, failure

class W100FanCluster(CustomCluster, Fan):
    """Fan cluster that proxies to W100 custom cluster."""

    _CONSTANT_ATTRIBUTES = {
        0x0001: Fan.FanModeSequence.Low_Med_High_Auto,
    }

    def __init__(self, *args, **kwargs):
        """Initialize the cluster."""
        super().__init__(*args, **kwargs)
        self._update_attribute(0x0000, Fan.FanMode.Auto)
        self._update_attribute(0x0001, Fan.FanModeSequence.Low_Med_High_Auto)

    async def write_attributes(self, attributes, manufacturer=None):
        """Intercept write attributes."""
        if "fan_mode" in attributes or 0x0000 in attributes:
            val = attributes.get("fan_mode", attributes.get(0x0000))
            
            # Map FanMode to W100 S
            # Auto(5) -> 0
            # Low(1) -> 1
            # Medium(2) -> 2
            # High(3) -> 3
            s = 0
            if val == Fan.FanMode.Low:
                s = 1
            elif val == Fan.FanMode.Medium:
                s = 2
            elif val == Fan.FanMode.High:
                s = 3
            
            _LOGGER.debug("W100FanCluster: Writing fan_mode %s -> S=%s", val, s)
            
            # Update Thermostat cache and send
            ep1 = self.endpoint.device.endpoints.get(1)
            if ep1:
                thermostat = ep1.in_clusters.get(Thermostat.cluster_id)
                if thermostat and isinstance(thermostat, W100ThermostatCluster):
                    thermostat._cached_s = s
                    
                    manu_cluster = ep1.in_clusters.get(W100ManuSpecificCluster.cluster_id)
                    if manu_cluster:
                        await manu_cluster.send_pmtsd(
                            thermostat._cached_p,
                            thermostat._cached_m,
                            thermostat._cached_t,
                            s,
                            thermostat._cached_d
                        )
            
            self._update_attribute(0x0000, val)
            
            records = [
                foundation.WriteAttributesStatusRecord(
                    foundation.Status.SUCCESS,
                    attrid=0x0000
                )
            ]
            return [records]
            
        return await super().write_attributes(attributes, manufacturer)


class W100ThermostatCluster(CustomCluster, Thermostat):
    """Thermostat cluster that proxies to W100 custom cluster."""

    _CONSTANT_ATTRIBUTES = {
        Thermostat.attributes_by_name["ctrl_sequence_of_oper"].id: Thermostat.ControlSequenceOfOperation.Cooling_and_Heating,
    }

    def __init__(self, *args, **kwargs):
        """Initialize the cluster."""
        super().__init__(*args, **kwargs)
        self._cached_p = 0
        self._cached_m = 0
        self._cached_t = 20.0
        self._cached_s = 0
        self._cached_d = 0
        
        # Initialize attributes
        self._update_attribute(0x0029, 0)  # running_state = Idle
        self._update_attribute(0x001C, Thermostat.SystemMode.Off)
        self._update_attribute(0x0012, 2000)
        self._update_attribute(0x0011, 2000)

    def recalculate_running_state(self):
        """Recalculate running state based on temp and setpoint."""
        local_temp = self.get(0x0000)
        mode = self.get(0x001C)
        
        if mode == Thermostat.SystemMode.Cool:
            setpoint = self.get(0x0011)
        else:
            setpoint = self.get(0x0012)

        if local_temp is None:
            temp_cluster = self.endpoint.in_clusters.get(TemperatureMeasurement.cluster_id)
            if temp_cluster:
                local_temp = temp_cluster.get(0x0000)
                if local_temp is not None:
                    self.update_attribute(0x0000, local_temp)

        if local_temp is None or setpoint is None or mode is None:
            return

        state = 0  # Idle
        if mode == Thermostat.SystemMode.Heat:
            if local_temp < setpoint:
                state = 0x0001 # Heat State On
        elif mode == Thermostat.SystemMode.Cool:
            if local_temp > setpoint:
                state = 0x0002 # Cool State On
        
        self.update_attribute(0x0029, state)

    async def write_attributes(self, attributes, manufacturer=None):
        """Intercept write attributes."""
        for attr, value in attributes.items():
            attr_id = attr
            if isinstance(attr, str):
                attr_id = self.attributes_by_name[attr].id

            # Update internal cache from write
            if attr in ("occupied_heating_setpoint", "occupied_cooling_setpoint") or attr_id in (0x0012, 0x0011):
                self._cached_t = value / 100.0
            
            if attr == "system_mode" or attr_id == 0x001C:
                mode = value
                if mode == Thermostat.SystemMode.Off:
                    self._cached_p = 1
                else:
                    self._cached_p = 0
                    if mode == Thermostat.SystemMode.Cool:
                        self._cached_m = 0
                    elif mode == Thermostat.SystemMode.Heat:
                        self._cached_m = 1
                    elif mode == Thermostat.SystemMode.Auto:
                        self._cached_m = 2
            
            self._update_attribute(attr_id, value)

        # Send update
        manu_cluster = self.endpoint.in_clusters.get(0xFCC0)
        if manu_cluster and isinstance(manu_cluster, W100ManuSpecificCluster):
            await manu_cluster.send_pmtsd(
                self._cached_p,
                self._cached_m,
                self._cached_t,
                self._cached_s,
                self._cached_d
            )
        
        records = [
            foundation.WriteAttributesStatusRecord(
                foundation.Status.SUCCESS,
                attrid=(self.attributes_by_name[attr].id if isinstance(attr, str) else attr)
            )
            for attr in attributes
        ]
        
        self.recalculate_running_state()
        return [records]


class XiaomiCustomDeviceV2(CustomDeviceV2):
    """Custom device representing xiaomi devices for V2 quirks."""

    def _find_zcl_cluster(
        self, hdr: foundation.ZCLHeader, packet: t.ZigbeePacket
    ) -> Cluster:
        """Find a cluster for the packet."""
        try:
            return super()._find_zcl_cluster(hdr, packet)
        except KeyError:
            endpoint = self.endpoints[packet.src_ep]
            if hdr.direction == foundation.Direction.Client_to_Server:
                return endpoint.in_clusters[packet.cluster_id]
            else:
                return endpoint.out_clusters[packet.cluster_id]


(
    QuirkBuilder("Aqara", "lumi.sensor_ht.agl001")
    .device_class(XiaomiCustomDeviceV2)
    .replaces(W100BasicCluster)
    .replaces(XiaomiPowerConfigurationPercent)
    .replaces(MultistateInputCluster, endpoint_id=1)
    .replaces(MultistateInputCluster, endpoint_id=2)
    .replaces(MultistateInputCluster, endpoint_id=3)
    .replaces(W100TemperatureMeasurement)
    .replaces(W100ManuSpecificCluster)
    .adds(W100ThermostatCluster)
    .adds_endpoint(21, profile_id=zha.PROFILE_ID, device_type=0x0301)
    .adds(W100FanCluster, endpoint_id=21)
    .enum(
        "sensor",
        W100ExternalSensorMode,
        cluster_id=W100ManuSpecificCluster.cluster_id,
        endpoint_id=1,
        translation_key="sensor",
        fallback_name="Sensor display mode",
    )
    .number(
        "external_temperature",
        cluster_id=W100ManuSpecificCluster.cluster_id,
        endpoint_id=1,
        min_value=-100,
        max_value=100,
        step=0.1,
        unit="°C",
        translation_key="external_temperature",
        fallback_name="External temperature",
    )
    .number(
        "external_humidity",
        cluster_id=W100ManuSpecificCluster.cluster_id,
        endpoint_id=1,
        min_value=0,
        max_value=100,
        step=0.1,
        unit="%",
        translation_key="external_humidity",
        fallback_name="External humidity",
    )
    .switch("auto_hide_middle_line", cluster_id=W100ManuSpecificCluster.cluster_id, endpoint_id=1, translation_key="auto_hide_middle_line", fallback_name="Auto Hide Middle Line", on_value=0, off_value=1)
    .switch("thermostat_mode_switch", cluster_id=W100ManuSpecificCluster.cluster_id, endpoint_id=1, translation_key="thermostat_mode_switch", fallback_name="Thermostat Mode Switch")
    .device_automation_triggers({
        (PLUS_BUTTON, SHORT_PRESS): {COMMAND: COMMAND_SINGLE, ENDPOINT_ID: 1},
        (PLUS_BUTTON, DOUBLE_PRESS): {COMMAND: COMMAND_DOUBLE, ENDPOINT_ID: 1},
        (PLUS_BUTTON, LONG_PRESS): {COMMAND: COMMAND_HOLD, ENDPOINT_ID: 1},
        (PLUS_BUTTON, LONG_RELEASE): {COMMAND: COMMAND_RELEASE, ENDPOINT_ID: 1},

        (CENTER_BUTTON, SHORT_PRESS): {COMMAND: COMMAND_SINGLE, ENDPOINT_ID: 2},
        (CENTER_BUTTON, DOUBLE_PRESS): {COMMAND: COMMAND_DOUBLE, ENDPOINT_ID: 2},
        (CENTER_BUTTON, LONG_PRESS): {COMMAND: COMMAND_HOLD, ENDPOINT_ID: 2},
        (CENTER_BUTTON, LONG_RELEASE): {COMMAND: COMMAND_RELEASE, ENDPOINT_ID: 2},

        (MINUS_BUTTON, SHORT_PRESS): {COMMAND: COMMAND_SINGLE, ENDPOINT_ID: 3},
        (MINUS_BUTTON, DOUBLE_PRESS): {COMMAND: COMMAND_DOUBLE, ENDPOINT_ID: 3},
        (MINUS_BUTTON, LONG_PRESS): {COMMAND: COMMAND_HOLD, ENDPOINT_ID: 3},
        (MINUS_BUTTON, LONG_RELEASE): {COMMAND: COMMAND_RELEASE, ENDPOINT_ID: 3},
    })
    .add_to_registry()
)

