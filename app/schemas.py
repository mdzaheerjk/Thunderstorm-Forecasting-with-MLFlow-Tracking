from pydantic import BaseModel
from typing import List

# to make sure we are accepting , what we should 

# structred format to accept input

class WeatherInput(BaseModel):
    SWEAT_index: float
    K_index: float
    Totals_totals_index: float
    Environmental_Stability: float
    Moisture_Indices: float
    Convective_Potential: float
    Temperature_Pressure: float
    Moisture_Temperature_Profiles: float
    
    def to_list(self):
        return [
            self.SWEAT_index,
            self.K_index,
            self.Totals_totals_index,
            self.Environmental_Stability,
            self.Moisture_Indices,
            self.Convective_Potential,
            self.Temperature_Pressure,
            self.Moisture_Temperature_Profiles
        ]
