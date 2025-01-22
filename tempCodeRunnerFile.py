from flask import Flask, request, jsonify
import requests
import pickle
import torch
import numpy as np
import pymongo
from sklearn.preprocessing import StandardScaler
import joblib
import torch
import torch.nn as nn  # This is where the nn module comes from
