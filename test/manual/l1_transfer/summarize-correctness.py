#!/usr/bin/env python3
"""Validate result completeness and summarize read-only NVLink/NIC counters."""
import argparse
import json
from pathlib import Path
import re


def nvlink(value):
    if value["nvlink"]["rc"]:
        raise ValueError("NVLink counter command failed")
    result = {}; gpu = None
    for line in value["nvlink"]["stdout"].splitlines():
        match = re.match(r"GPU (\d+):", line)
        if match:
            gpu = match[1]; result[gpu] = {"Tx":0,"Rx":0}
        match = re.search(r"Link \d+: Data (Tx|Rx): (\d+) KiB", line)
        if match and gpu is not None:
            result[gpu][match[1]] += int(match[2])*1024
    if not result: raise ValueError("missing NVLink counters")
    return result


def delta(before, after, data_hcas):
    b, a = nvlink(before), nvlink(after)
    if b.keys()!=a.keys(): raise ValueError("GPU counter set changed")
    links = {gpu:{direction:a[gpu][direction]-b[gpu][direction] for direction in ("Tx","Rx")} for gpu in b}
    if before["rdma"].keys()!=after["rdma"].keys(): raise ValueError("NIC counter set changed")
    nics = {}
    for path, value in before["rdma"].items():
        pieces=Path(path).parts; hca=pieces[pieces.index("infiniband")+1]
        name=pieces[-1]; nics.setdefault(hca,{"port_xmit_data":0,"port_rcv_data":0})
        nics[hca][name] += (after["rdma"][path]-value)*4 # IB port_*_data units are 4 octets.
    data_total = sum(sum(nics[hca].values()) for hca in data_hcas if hca in nics)
    if not set(data_hcas).issubset(nics): raise ValueError("missing configured data HCA counters")
    if any(v<0 for counters in (*links.values(),*nics.values()) for v in counters.values()):
        raise ValueError("counter reset/wrap detected")
    return {"nvlink_bytes":links,"nic_bytes":nics,"data_hca_total_bytes":data_total}


def summarize(root, data_hcas):
    rows=[]
    for topology in ("local","cross-node"):
        base=root/topology
        owner=json.loads((base/"rank-0/owner-result.json").read_text())
        consumer=json.loads((base/"rank-1/consumer-result.json").read_text())
        if not owner["pass"] or not consumer["pass"] or consumer["forbidden_imports"]:
            raise ValueError("GPU correctness or independence failed")
        if (owner["host"] == consumer["host"]) != (topology == "local"):
            raise ValueError("topology does not match requested test")
        expected={(case,op) for case in ("odd-boundaries","contiguous","fragmented-128","layered-kv-ranges","planned-same-layout") for op in ("READ","WRITE")}
        for result in (owner,consumer):
            if len(result["cases"])!=len(expected) or {(r["case"],r["opcode"]) for r in result["cases"]}!=expected:
                raise ValueError("missing/duplicate correctness cases")
        owner_rows={(r["case"],r["opcode"]):r for r in owner["cases"]}
        for row in consumer["cases"]:
            peer=owner_rows[row["case"],row["opcode"]]
            if not row["pass"] or not peer["pass"] or (row["bytes"],row["ranges"]) != (peer["bytes"],peer["ranges"]):
                raise ValueError("inconsistent producer/consumer results")
            item={"topology":topology,**row,"owner_host":owner["host"]}
            for rank,role in ((0,"owner"),(1,"consumer")):
                paths=[base/f"rank-{rank}"/f"{row['case']}-{row['opcode']}-{role}-{phase}.json" for phase in ("before","after")]
                item[role+"_counters"]=delta(*(json.loads(p.read_text()) for p in paths),data_hcas)
            rows.append(item)
    return {"pass":True,"gpu_cases":len(rows),"data_hcas":data_hcas,"cases":rows,
            "note":"KiB counters have per-link quantization; small/unaligned payloads need not equal link bytes. No performance claim."}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("run",type=Path)
    parser.add_argument("--data-hcas",required=True)
    args=parser.parse_args()
    report=summarize(args.run,args.data_hcas.split(","))
    (args.run/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print("PASS",report["gpu_cases"],"GPU cases")
    for row in report["cases"]:
        if row["case"]=="contiguous":
            print(row["topology"],row["opcode"],"payload",row["bytes"],
                  "owner",row["owner_counters"]["nvlink_bytes"],
                  "consumer",row["consumer_counters"]["nvlink_bytes"],
                  "data NIC bytes",row["owner_counters"]["data_hca_total_bytes"],row["consumer_counters"]["data_hca_total_bytes"])
