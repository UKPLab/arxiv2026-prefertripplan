import pandas as pd
from pathlib import Path
from pandas import DataFrame
from typing import Optional
from utils.func import extract_before_parenthesis


# Repo root: travelplanner-ports/tools/accommodations/apis.py -> ../../..
_DEFAULT_PATH = (Path(__file__).resolve().parents[3]
                  / "database" / "accommodations"
                  / "clean_accommodations_2025.csv")


class Accommodations:
    # Only NAME + city are load-bearing (they're what every lookup keys
    # on); every other column is allowed to be null.  Original
    # TravelPlanner used a blanket `.dropna()` which silently evicted
    # legitimate listings whenever any single cell was empty.
    _REQUIRED_COLUMNS = ['NAME', 'city']

    def __init__(self, path=str(_DEFAULT_PATH)):
        self.path = path
        self.data = pd.read_csv(self.path).dropna(subset=self._REQUIRED_COLUMNS)[['NAME','price','room type', 'house_rules', 'minimum nights', 'maximum occupancy', 'review rate number', 'city']]
        print("Accommodations loaded.")

    def load_db(self):
        self.data = pd.read_csv(self.path).dropna()

    def run(self,
            city: str,
            ) -> DataFrame:
        """Search for accommodations by city."""
        results = self.data[self.data["city"] == city]
        if len(results) == 0:
            return "There is no attraction in this city."
        
        return results
    
    def run_for_annotation(self,
            city: str,
            ) -> DataFrame:
        """Search for accommodations by city."""
        results = self.data[self.data["city"] == extract_before_parenthesis(city)]
        return results