import joblib
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from werkzeug.security import generate_password_hash, check_password_hash
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import datetime
import os

from utility.intent import handle_input, get_intent

# region  Flask Configuration
app = Flask(__name__)

# Configuration
app.config.from_object('config.Config')

# Initialize extensions
db = SQLAlchemy(app)

migrate = Migrate(app, db)
jwt = JWTManager(app)

from models import User
models_path = './models/'
# Load the pre-trained models (already trained using Grid Search)
knn_model = joblib.load(models_path + 'best_knn_model.pkl')
lr_model = joblib.load(models_path + 'best_lr_model.pkl')
dt_model = joblib.load(models_path + 'best_dt_model.pkl')

chats = {}


# endregion

# region  User signup and login
@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if User.query.filter_by(username=username).first():
        return jsonify({"message": "User already exists"}), 400

    hashed_password = generate_password_hash(password)
    new_user = User(username=username, password=hashed_password)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({"message": "User created successfully"}), 201


# User login route
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    user = User.query.filter_by(username=username).first()

    if not user or not check_password_hash(user.password, password):
        return jsonify({"message": "Invalid username or password"}), 401

    access_token = create_access_token(identity=user.id, expires_delta=datetime.timedelta(hours=1))
    return jsonify(access_token=access_token), 200


# Protected route
@app.route('/protected', methods=['GET'])
@jwt_required()
def protected():
    user_id = get_jwt_identity()
    return jsonify(logged_in_as=user_id), 200
# endregion


# region Chat API
@app.route("/get_intent", methods=['POST'])
@jwt_required()
def intent():
    data = request.get_json()
    chat_id = data.get('chat_id')
    if chat_id is not None:
        intent = get_intent(data['message'])
        if intent == 'symptoms':
            if chats.get(chat_id) is not None:
                chats["messages"].append(data['message'])
            else:
                chats[chat_id] = {
                    "messages": [],
                    "symptoms": []
                }
        return jsonify({"intent": intent, "msg": 'got intent of message'})
    else:
        return jsonify({
            'error': "No chat id provided",
        }, status=422)


@app.route('/predict', methods=['POST'])
def chat():
    pass
# endregion
