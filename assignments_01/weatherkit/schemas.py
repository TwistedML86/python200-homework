from pydantic import BaseModel, Field, model_validator

class HourlyBlock(BaseModel):
    """Stores hourly times, temperatures in Celsius, and precipitation in mm."""
    time:list [str]
    temperature_2m: list[float]
    precipitation: list[float]

    @model_validator(mode="after")
    def check_lengths(self):
        """Checks that the three lists have the same length."""
        if not (
            len(self.time)
            == len(self.temperature_2m)
            == len(self.precipitation)

        ):
            raise ValueError("Hourly lists must have the same length")
        return self

class WeatherResponse(BaseModel):
    """Stores the location information and hourly weather data."""

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    timezone: str
    elevation: float
    hourly: HourlyBlock
