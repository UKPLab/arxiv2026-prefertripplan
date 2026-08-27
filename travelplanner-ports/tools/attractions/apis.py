import pandas as pd
from pathlib import Path
from pandas import DataFrame
from typing import Optional
from utils.func import extract_before_parenthesis


_DEFAULT_PATH = (Path(__file__).resolve().parents[3]
                  / "database" / "attractions" / "attractions.csv")


class Attractions:
    # Only Name + City are load-bearing (they're what every lookup keys
    # on); other columns are allowed to be null.
    _REQUIRED_COLUMNS = ['Name', 'City']

    def __init__(self, path=str(_DEFAULT_PATH)):
        self.path = path
        # PreferTripPlan additions: `categories` (list-literal string) and
        # `rating` (float) live in this project's attractions.csv but
        # were not part of TravelPlanner's original schema.  We keep the
        # TravelPlanner columns and append the new two, and drop `Phone`
        # which is absent from our CSV.
        self.data = pd.read_csv(self.path).dropna(subset=self._REQUIRED_COLUMNS)[
            ['Name', 'Latitude', 'Longitude', 'Address', 'Website',
             'City', 'categories', 'rating']
        ]
        print("Attractions loaded.")

    def load_db(self):
        self.data = pd.read_csv(self.path)

    def run(self,
            city: str,
            ) -> DataFrame:
        """Search for Accommodations by city and date."""
        results = self.data[self.data["City"] == city]
        # the results should show the index
        results = results.reset_index(drop=True)
        if len(results) == 0:
            return "There is no attraction in this city."
        return results  
      
    def run_for_annotation(self,
            city: str,
            ) -> DataFrame:
        """Search for Accommodations by city and date."""
        results = self.data[self.data["City"] == extract_before_parenthesis(city)]
        # the results should show the index
        results = results.reset_index(drop=True)
        return results