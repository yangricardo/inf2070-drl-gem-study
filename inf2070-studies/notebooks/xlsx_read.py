import pandas as pd
import datetime
from logging import getLogger

logger = getLogger(__name__)

class XlsxReader:
    """
    A class to read data from an Excel file (.xlsx) and return it as a list of dictionaries.
    Each dictionary corresponds to a row in the Excel file, with keys as column headers.
    """

    def __init__(self, filename):
      start_time = datetime.datetime.now()
      logger.info(f"Reading Excel file: {filename}")
      self.filename = filename
      self.xlsx = pd.ExcelFile(filename)
      self.sheet_names = self.xlsx.sheet_names
      self.columns: list[str] = {}
      self.data: dict[str, list[dict]] = {}
      read_futures = {sheet: self.read_sheet(sheet_name=sheet) for sheet in self.sheet_names}
      logger.info(f"Reading {len(read_futures)} sheets from the Excel file.")
      logger.info(f"Reading sheets: {';'.join(self.sheet_names)}")
      for sheet, future in read_futures.items():
        try:
          future.result()          
        except Exception as e:
          logger.error(f"Error reading sheet '{sheet}': {e}")

      end_time = datetime.datetime.now()
      logger.info(f"XlsxReader initialized in {end_time - start_time} seconds.")

    def read_sheet(self, sheet_name):
      """
      Read a specific sheet from the Excel file and return it as a list of dictionaries.
      Each dictionary corresponds to a row in the sheet, with keys as column headers.
      """
      logger.info(f"Reading sheet '{sheet_name}' from the Excel file.")
      if sheet_name not in self.sheet_names:
        logger.error(f"Sheet '{sheet_name}' does not exist in the Excel file.")
        return None
      if sheet_name in self.data:
        logger.info(f"Using cached data for sheet '{sheet_name}'.")
        return self.df[sheet_name]
      df = pd.read_excel(self.xlsx, sheet_name=sheet_name)
      self.columns[sheet_name] = df.columns.tolist()
      self.data[sheet_name] = df.to_dict(orient='records')
      logger.info(f"Sheet '{sheet_name}' {self.data[sheet_name].__len__()} records and {self.columns[sheet_name].__len__()}  columns...")
      return self.data[sheet_name]

    