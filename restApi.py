
import logging
from logging.handlers import RotatingFileHandler
import os
import time

from flask import Flask

logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger(__name__)

# Setup file handler
file_handler = RotatingFileHandler(
    "./logs/logfile.log", maxBytes=1024 * 1024 * 100, backupCount=20
)

app = Flask(__name__)
app.logger.addHandler(file_handler)

@app.route("/get_episode_path", methods=["GET"])
def get_episode_path():
    while True:
        try:
            boosted_episodes = []  # For donations and such to play the episode quickly
            unreleased_episodes = boosted_episodes + os.listdir(".shared/unreleased_episodes/")
            released_episodes = os.listdir(".shared/released_episodes/")


            if len(unreleased_episodes) == 0 and len(released_episodes) == 0:
                while len(os.listdir(".shared/unreleased_episodes/")) == 0:
                    logging.debug("Waiting to generatee content. Sleeping for 10 seconds...")
                    time.sleep(10)
            
            if len(unreleased_episodes) == 0:
                logging.debug("Fallback to replaying old episode.")
                episode_to_release = random.choice(released_episodes)
                shutil.copytree(
                    f".shared/released_episodes/{episode_to_release}",
                    f".shared/unreleased_episodes/{episode_to_release}",
                )
                shutil.rmtree(f".shared/released_episodes/{episode_to_release}")
                unreleased_episodes = os.listdir(".shared/unreleased_episodes/")

            episode_to_release = random.choice(unreleased_episodes[:3])
            logging.info(f"Selected episode for release: {episode_to_release}")

            unity_episode_path = os.path.join(streaming_assets_path, episode_to_release)

            if os.path.exists(unity_episode_path):
                shutil.rmtree(unity_episode_path)
                

            shutil.copytree(
                ".shared/unreleased_episodes/" + episode_to_release, unity_episode_path
            )
            shutil.copytree(
                ".shared/unreleased_episodes/" + episode_to_release,
                ".shared/released_episodes/" + episode_to_release,
            )
            shutil.rmtree(".shared/unreleased_episodes/" + episode_to_release)

            app.logger.info(f"Released episode: {episode_to_release}")
            return jsonify({"episode_path": streaming_assets_path + episode_to_release})
        except Exception as e:
            logger.exception(f"An error occurred: {e}")
            time.sleep(2)
            # return f"A python-side error ocurred: {e}"


@app.route("/set_supported_scenes", methods=["POST"])
def set_supported_scenes():
    global supported_scenes
    data = request.json
    print(data)
    supported_scenes = SupportedScenes.from_json(json.dumps(data))
    

    return "Supported scenes set successfully!"


if __name__ == "__main__":
    # Start the background task
    thread = threading.Thread(target=generate_episodes)
    thread.daemon = (
        True  # This ensures the thread will be killed when the main thread is killed
    )
    thread.start()
    app.run(debug=False, host="localhost", port=5000)
