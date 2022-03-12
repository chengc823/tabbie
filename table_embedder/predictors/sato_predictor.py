
import copy
import os
import json
import numpy as np
from pathlib import Path
from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register('sato_predictor')
class CellPredictor(Predictor):
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        line = json_dict
        # idx = line['old_id'] if 'old_id' in line.keys() else line['id']
        idx = line['table_id'] if 'table_id' in line.keys() else None
        fname = line['locator'] if 'locator' in line.keys() else None

        max_rows, max_cols = 30, 20
        table_data = line['table_data']
        table = copy.deepcopy(table_data)
        table_header = table_data[0]
        table_data = table_data[1:max_rows]
        if len(table_data[0]) > max_cols:
            table_header = np.array(table_header)[:max_cols].tolist()
            table_data = np.array(table_data)[:, :max_cols].tolist()
        n_cols = len(table_data[0])
        n_rows = len(table_data)
        # cell_labels = line['cell_labels'] if 'cell_labels' in line else None
        # col_labels = line['col_labels'] if 'col_labels' in line else None
        # table_labels = line['table_labels'] if 'table_labels' in line else None
        label_idx = line['label_idx'] if 'label_idx' in line.keys() else None
        col_idx = line['col_idx'] if 'col_idx' in line.keys() else None 

        instance = self._dataset_reader.text_to_instance(
            table_id=idx,
            table_header=table_header,
            n_cols = n_cols,
            n_rows = n_rows,
        #    cell_labels=cell_labels,
            label_idx=label_idx,
            col_idx=col_idx,
        #    table_labels=table_labels,
            table_data=table_data,
        #    table=table
            fname=fname 
        )
        return instance





