import torch
import torch.nn as nn


class VisualEncoder(nn.Module):
    def __init__(self, app_feat, mot_feat, app_input_size, mot_input_size, app_output_size, mot_output_size,
            ss_feat, ss_input_size, ss_output_size):
        super(VisualEncoder, self).__init__()
        self.app_feat = app_feat
        self.mot_feat = mot_feat
        self.app_input_size = app_input_size
        self.mot_input_size = mot_input_size
        self.app_output_size = app_output_size
        self.mot_output_size = mot_output_size

        self.ss_feat = ss_feat
        self.ss_input_size = ss_input_size
        self.ss_output_size = ss_output_size
        assert self.ss_output_size == self.mot_input_size, \
        'ss_output_size: {} does not math mot_input_size: {}'.format(self.ss_output_size, self.mot_input_size)

        self.app_linear = nn.Linear(self.app_input_size, self.app_output_size)
        self.mot_linear = nn.Linear(self.mot_input_size, self.mot_output_size)
        self.ss_linear = nn.Linear(self.ss_input_size, self.ss_output_size)
        self.relu = nn.ReLU()

    def forward(self, app_feats, mot_feats, ss_feats):
        ss_outputs = self.relu(self.ss_linear(ss_feats))
        mot_feats = mot_feats + 0.25*ss_outputs
        app_outputs = self.app_linear(app_feats)
        mot_outputs = self.mot_linear(mot_feats)
        
        return torch.cat([ app_outputs, mot_outputs ], dim=2)

