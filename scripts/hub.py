import json
from time import time, sleep
from threading import Thread
from socket import socket, AF_INET, SOCK_DGRAM, SOL_SOCKET, SO_BROADCAST, gethostbyname, gethostname
from uuid import uuid1
import binascii
import shelve
import logging

from flask import Flask, render_template, jsonify, request, abort
import schedule
import yaml
import rsa
import requests
from itsdangerous import Signer, TimedSerializer

import mechwolf as mw

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("schedule").setLevel(logging.WARNING)
logging.getLogger("werkzeug").setLevel(logging.INFO)

app = Flask(__name__) # create flask app

# how long to wait for check ins before aborting a protcol
TIMEOUT = 60

# get the config data
with open("hub_config.yml", "r") as f:
    config = yaml.load(f)
SECURITY_KEY = config['resolver_info']['security_key']
HUB_ID = config['resolver_info']['hub_id']
signer, serializer = Signer(SECURITY_KEY), TimedSerializer(SECURITY_KEY)

# get the private key
with open(config['resolver_info']['rsa_private_filepath'], mode='rb') as privatefile:
    keydata = privatefile.read()
PRIV_KEY = rsa.PrivateKey.load_pkcs1(keydata)

# get the public key
with open(config['resolver_info']['rsa_public_filepath'], mode='rb') as pubfile:
    keydata = pubfile.read()
# the inverse function call here is to verify that they public key is valid
PUB_KEY_HEX = binascii.hexlify(rsa.PublicKey.load_pkcs1(keydata).save_pkcs1()).decode()

def update_ip():
    '''send out the location of our server to the resolver if nescessary '''

    # find current IP address
    ip_socket = socket(AF_INET, SOCK_DGRAM)
    ip_socket.connect(("8.8.8.8", 80))
    my_ip = ip_socket.getsockname()[0]
    ip_socket.close()

    with shelve.open('hub_config') as config_db:
        # return early if address on file matches
        try:
                if my_ip == config_db["current_ip"]:
                    logging.debug("No change to IP")
                    return
        except KeyError:
            pass

        # store new IP address
        config_db["current_ip"] = my_ip

    # sign the current IP address
    signed_ip = signer.sign(my_ip.encode())
    signature = rsa.sign(signed_ip, PRIV_KEY, 'SHA-512')
    signature = binascii.hexlify(signature).decode()

    # send it to the resolver
    payload = {"hub_id": HUB_ID,
               "hub_address": signed_ip,
               "hub_address_signature": signature,
               "hub_public_key": PUB_KEY_HEX}
    requests.post(mw.RESOLVER_URL + "register", data=payload)

    logging.info(f"Updated resolver with IP {my_ip}.")

def run_schedule():
    while True:
        schedule.run_pending()
        sleep(2.5)

@app.route("/submit_protocol", methods=["POST"])
def submit_protocol():
    '''Accepts a protocol posted as JSON.'''
    logging.info("Recieved protocol")
    try:
        protocol = serializer.loads(request.form.get("protocol_json"), max_age=5)
        logging.debug("Protocol signature is valid")
    except:
        logging.warning("Protocol signature is invalid!")
        return "protocol rejected: invalid signature"

    with shelve.open('hub_shelf') as db:
        db["protocol"] = protocol
        db["protocol_id"] = str(uuid1())

        # clear the stored values when a new protocol comes in
        db["protocol_acks"] = set()
        db["start_time_acks"] = set()
        try:
            del db["start_time"]
        except KeyError:
            pass

        # store the time when the protocol came in
        db["protocol_submit_time"] = time()
        return jsonify(dict(protocol_id=db["protocol_id"]))

@app.route("/protocol", methods=["GET", "POST"])
def protocol():
    '''Returns protocols, if availible.'''
    try:
        with shelve.open('hub_shelf') as db:

            # load the protocol and add the protocol_id
            parsed_protocol = json.loads(db["protocol"])
            parsed_protocol.update({"protocol_id": db["protocol_id"]})

            # to allow easier introspection, let people view the protocol
            if request.method == "GET":
                return jsonify(parsed_protocol)

            # only give the protocol once
            device_id = request.form["device_id"]
            try:
                if device_id in db["protocol_acks"]:
                    return "no protocol"
            except KeyError:
                pass

            # store the device that checked in and and return the protocol
            db["protocol_acks"] = db["protocol_acks"].union([device_id])

            return app.response_class(
                response=json.dumps({k: parsed_protocol[k] for k in ["protocol_id", device_id]}),
                status=200,
                mimetype="application/json")

    # if no protocol has been given
    except KeyError:
        return "no protocol"

@app.route("/start_time")
def start_time():
    with shelve.open('hub_shelf') as db:
        try:
            # time out if too long has passed from when the protocol was submitted but not all devices have checked in
            logging.debug("Checking to see if timing out...")
            if time() - float(db["protocol_submit_time"]) > TIMEOUT:
                return "abort"
            logging.debug("Not timing out")

            # if every device has gotten the protocol, give them the start time
            logging.debug(f'Checking if all of {db["protocol_acks"]} are in {list(json.loads(db["protocol"]))}.')
            if all([x in db["protocol_acks"] for x in list(json.loads(db["protocol"]))]):
                logging.debug("They are!")

                # log the device ID as having gotten start time
                if request.args.get("device_id") in db["start_time_acks"]:
                        return "no start time"
                elif request.args.get("device_id") is not None:
                    db["start_time_acks"] = db["start_time_acks"].union([request.args.get("device_id")])

                logging.debug(f"List of acknowledged start times is now {db['start_time_acks']}")

                # the first device after all have checked in will determine start time
                try:
                    return str(db["start_time"])
                except KeyError:
                    if request.args.get("device_id") is not None: # return "no start time" if a blank request is gotten
                        logging.debug("No start time set. Setting new one...")
                        db["start_time"] = time() + 5
                        logging.debug(f'Start time is {db["start_time"]}')
                        return str(db["start_time"])

        except KeyError:
            pass
        return "no start time"

@app.route("/log", methods=["POST", "GET"])
def log():
    logging.info(f"Logging {request.json}")
    with shelve.open('hub_shelf') as db:
        protocol_id = db["protocol_id"]
    with shelve.open(protocol_id) as db:
        if request.method == "GET":
            try:
                return str(db["log"])
            except KeyError:
                return "no log"
        try:
            db["log"] = db["log"] + [request.json]
        except KeyError:
            db["log"] = [request.json]
    return "logged"


schedule.every(5).seconds.do(update_ip)
t = Thread(target=run_schedule)
t.start()
app.run(debug=True, host="0.0.0.0", use_reloader=True, threaded=True, port=80)
