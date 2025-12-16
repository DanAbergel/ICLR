# import torch.nn as nn
#
#
# class PredictionHead(nn.Module):
#     def __init__(self, embedding_dim, output_dims, p_dropout=0.2):
#         super().__init__()
#
#         self.input_dim = embedding_dim
#         self.hidden_dim1 = embedding_dim // 2
#         # self.hidden_dim2 = embedding_dim // 4
#
#         self.task_heads = nn.ModuleDict()
#         for task, output_dim in output_dims.items():
#             self.task_heads[task] = nn.Sequential(
#                 nn.Linear(self.input_dim, self.hidden_dim1),
#                 nn.GELU(),
#                 nn.Dropout(p_dropout),
#                 nn.Linear(self.hidden_dim1, self.hidden_dim2),
#                 nn.GELU(),
#                 # nn.Dropout(p_dropout),
#                 # nn.Linear(self.hidden_dim2, output_dim),
#             )
#
#     def forward(self, x):
#         task_predictions = {task: head(x) for task, head in self.task_heads.items()}
#         return task_predictions


import torch.nn as nn

class PredictionHead(nn.Module):
    def __init__(self, embedding_dim, output_dims, p_dropout=0.5):
        super().__init__()

        hidden_dim = 64  # volontairement petit

        self.task_heads = nn.ModuleDict()
        for task, output_dim in output_dims.items():
            self.task_heads[task] = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p_dropout),
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(self, x):
        return {task: head(x) for task, head in self.task_heads.items()}