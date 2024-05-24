import logging
import os

LICENSE_URL = "https://github.com/FraunhoferMEVIS/MedicalMultitaskModeling/blob/main/license.md"
LICENSE_ENV_VAR_NAME = "MMM_LICENSE_ACCEPTED"
ACCEPTANCE_VALUE = "i accept"
ACCEPTANCE_PROMPT = f"Type {ACCEPTANCE_VALUE} to accept the MMM license agreement at {LICENSE_URL}:"
LICENSE_ACCEPT_INSTRUCTION = (
    f"Please accept the license agreement at {LICENSE_URL}"
    + f" by setting the environment variable {LICENSE_ENV_VAR_NAME} to '{ACCEPTANCE_VALUE}'."
)


def promptable():
    try:
        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"  # type: ignore
    except:
        return False


def verify_license_accepted():
    if LICENSE_ENV_VAR_NAME not in os.environ or os.environ[LICENSE_ENV_VAR_NAME].lower() != ACCEPTANCE_VALUE:
        logging.error(LICENSE_ACCEPT_INSTRUCTION)

        if promptable():
            # Enable the user to accept the license agreement interactively
            user_input = input(ACCEPTANCE_PROMPT)
            if user_input.lower() != ACCEPTANCE_VALUE:
                raise Exception("You must accept the license agreement to proceed.")
            else:
                os.environ[LICENSE_ENV_VAR_NAME] = ACCEPTANCE_VALUE
            return verify_license_accepted()
        raise Exception(LICENSE_ACCEPT_INSTRUCTION)


verify_license_accepted()
