# Aqara W100 (lumi.sensor_ht.agl001) ZHA Quirk

This custom quirk enables functionality for the Aqara W100 Climate Sensor in Home Assistant using ZHA (Zigbee Home Automation). It is mainly used for debugging and testing. By enabling Thermostat Mode Switch it is possible to control the middle display.

## Features

- **Thermostat Control:** Full support for Heating, Cooling, Auto, and Off modes.
- **Fan Control:** Dedicated Fan entity (Endpoint 21) supporting Low, Medium, High, and Auto speeds.
- **Sensors:** Temperature, Humidity, and Battery reporting.
- **Button Actions:** Exposes the 3 buttons (Plus, Center, Minus) as device automation triggers (Single, Double, Hold, Release). **NOTE**, if the Thermostat Mode Switch is enabled, then it is not possible for button action as the button control the display.
- **Configuration:**
  - **Thermostat Mode Switch:** Toggle the device's internal thermostat logic.
  - **Auto Hide Middle Line:** Control the display behavior.

## Installation

1. **Prepare the Directory:**
   - Access your Home Assistant configuration directory (where `configuration.yaml` is located).
   - Create a folder named `custom_zha_quirks` (if it doesn't exist).

2. **Copy the Quirk:**
   - Download the `sensor_ht_agl001.py` file.
   - Place it inside the `custom_zha_quirks` folder.

3. **Update Configuration:**
   - Edit your `configuration.yaml` file to include the custom quirks path:
     ```yaml
     zha:
        custom_quirks_path: /config/custom_zha_quirks/ 
     ```

4. **Restart Home Assistant:**
   - Restart Home Assistant to load the new configuration.

5. **Pair/Re-pair Device:**
   - If the device is already paired, you may need to remove it and pair it again for the quirk to apply correctly and create all entities (especially the new Fan entity).

## Usage

### Fan Control
The quirk creates a separate Fan entity.
- **Low / Medium / High:** Maps directly to the device's fan speeds.
- **Off:** Maps to the device's "Auto" fan mode (internally handled as Speed 0).

### Thermostat
The main climate entity controls the device's setpoint and system mode.
- **Heat/Cool/Auto:** Syncs with the device's internal logic.
- **Off:** Turns off the thermostat display/logic on the device.

### Buttons
You can use the buttons in Home Assistant Automations.
- Triggers: `Plus Button Short Press`, `Center Button Double Press`, etc.

## Troubleshooting

- **Entities missing?** Try removing the device from ZHA and pairing it again.
- **Logs:** Check Home Assistant logs for `zhaquirks.xiaomi.aqara.sensor_ht_agl001` for debug information.
