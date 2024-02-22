import os
import requests
import json


def get_nomad_allocations():
    url = os.getenv("NOMAD_ADDR") + "v1/job/" + os.getenv("NOMAD_JOB_ID") + "/allocations"
    response = requests.get(url, verify=os.getenv("NOMAD_CACERT"))
    return response.json()


def get_connection_of_rank(portname: str) -> dict[int, tuple[str, int]]:
    """
    Return a dictionary with the rank as key and a tuple with the address and the port number as value.
    """
    allocs = get_nomad_allocations()
    res: dict[int, tuple[str, int]] = {}
    for alloc_overview in allocs:
        # return alloc_overview
        if alloc_overview["ClientStatus"] != "running":
            continue
        alloc = requests.get(
            f"{os.getenv('NOMAD_ADDR')}/v1/allocation/{alloc_overview['ID']}",
            verify=os.getenv("NOMAD_CACERT"),
        ).json()
        alloc_name = alloc["Name"]
        rank = int(alloc_name[alloc_name.rfind("[") + 1 : alloc_name.rfind("]")])
        allports = alloc["Resources"]["Networks"][0]["DynamicPorts"]
        port = [p for p in allports if p["Label"] == portname][0]["Value"]
        res[rank] = alloc["Resources"]["Networks"][0]["IP"], port
    return res
