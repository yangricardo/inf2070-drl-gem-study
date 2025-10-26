import os
import pandas as pd
from datetime import datetime

from logging import getLogger
logger = getLogger(__name__)

class XlsxGenerator:
  def __init__(self, output_path: str, timestamp_suffix: bool = True):
    self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_path = output_path.replace('.xlsx', '')
    if timestamp_suffix:
      output_path = f"{output_path}_{self.timestamp}"
    output_path = f"{output_path}.xlsx"
    # Convert relative path to absolute path if needed
    if not os.path.isabs(output_path):
      output_path = os.path.abspath(os.path.join(os.getcwd(), output_path))
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Store the processed path
    self.output_path = output_path
    self.writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
    
  def add_dataframes(self, sheet_name: str, data: list[dict] | dict | None, columns: list[str] | None = None):
    if data is not None:
      if isinstance(data, dict):
        data = [data]
      elif isinstance(data, list) and data:
        data = data
      else:
        data = []
    if columns is None or columns == []:
      columns = list(data[0].keys()) if data else []
    df = pd.DataFrame(data, columns=columns)
    df = df.dropna(axis=1, how='all')
    sheet_name = sheet_name[:31]
    df.to_excel(self.writer, sheet_name=sheet_name, index=False)
    self.format_spreadsheet(self.writer.sheets[sheet_name], df)

  def format_spreadsheet(self, worksheet, df: pd.DataFrame):
    wrap_format = self.writer.book.add_format({'text_wrap': True})
    header_format = self.writer.book.add_format({
        'bold': True,
        'text_wrap': True,
        'bg_color': '#404040',
        'font_color': '#FFFFFF',
        'align': 'center',
        'valign': 'vcenter',
        'border': 1
    })
    stripe_format_1 = self.writer.book.add_format({
        'bg_color': '#F2F2F2',
        'text_wrap': True,
        'border': 1
    })
    stripe_format_2 = self.writer.book.add_format({
        'bg_color': '#E0E0E0',
        'text_wrap': True,
        'border': 1
    })

    for i, col in enumerate(df.columns):
      max_len = max(
        df[col].astype(str).map(len).max() if len(df) > 0 else 0,
        len(str(col))
      ) + 2
      worksheet.set_column(i, i, min(max_len, 70), wrap_format)

    # Enable autofilter for all columns in the header row (do this first)
    worksheet.autofilter(0, 0, len(df), len(df.columns) - 1)

    # Apply header format to each header cell individually
    for col_idx in range(len(df.columns)):
      worksheet.write(0, col_idx, df.columns[col_idx], header_format)

    worksheet.freeze_panes(1, 0)

    for row_idx in range(1, len(df) + 1):
      fmt = stripe_format_1 if row_idx % 2 == 1 else stripe_format_2
      worksheet.set_row(row_idx, None, fmt)

  def save(self):
    self.writer.close()
    # Save the file
    logger.info(f"File saved at: {self.output_path}")
    # copy generated file to output_path without timestamp
    base_output_path = self.output_path.replace(f"_{self.timestamp}", "")
    if base_output_path != self.output_path:
      try:
        import shutil
        shutil.copy2(self.output_path, base_output_path)
        logger.info(f"File copied to: {base_output_path}")
      except Exception as e:
        logger.error(f"Failed to copy file to {base_output_path}: {e}")
    return self.output_path