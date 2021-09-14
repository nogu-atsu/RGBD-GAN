# !/usr/bin/env python
# -*- coding: utf-8 -*-

import yaml


# Copy from tgans repo.
class Config(object):
    def __init__(self, config_dict):
        self.config = config_dict

    def __getattr__(self, key):
        if key in self.config:
            return self.config[key]
        else:
            return None

    def __getitem__(self, key):
        return self.config[key]

    def __repr__(self):
        return yaml.dump(self.config, default_flow_style=False)
