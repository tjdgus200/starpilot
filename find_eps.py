#!/usr/bin/env python3
import time
import sys
from panda import Panda

def find_eps():
  try:
    print("Connecting to Panda...")
    p = Panda()
    print(f"Connected to Panda ({p.get_serial()[0]})")
  except Exception as e:
    print(f"Error connecting to Panda: {e}")
    print("ERROR: Make sure OpenPilot is STOPPED before running this script!")
    print("Run: tmux kill-server  (or kill active processes)")
    return

  # Safety setup - allow ELM327/Diagnostics on Bus 0
  # We set safety to ALLOUTPUT to allow sending arbitrary UDS messages
  # WARNING: Do not drive while running this script!
  p.set_safety_mode(Panda.SAFETY_ALLOUTPUT)

  # Standard GM UDS IDs to scan
  # 0x24B: Camera (Known)
  # 0x241: EBCM (Brake) - Likely
  # 0x242: PSCM (Steering) - Suspected
  # 0x243-0x24F: Others
  # 0x7E0: Engine/ECM (Standard OBD)
  target_ids = [0x241, 0x242, 0x243, 0x244, 0x24B, 0x7E0]
  
  print("\nScanning for ECUs on Bus 0 (PT)...")
  
  found_eps = False

  for tx_id in target_ids:
    rx_id = tx_id + 0x400 # GM Logic: Response = Request + 0x400
    
    # UDS: Tester Present (0x3E 00)
    # 0x02 = Length, 0x3E = SID, 0x00 = Subfunc, 00 00 00 00 00 = Padding
    msg = b'\x02\x3E\x00\x00\x00\x00\x00\x00'
    
    # Send to Bus 0
    p.can_send(tx_id, msg, bus=0)
    time.sleep(0.1) # Wait for response
    
    # Check buffer
    can_recv = p.can_recv()
    response_found = False
    
    for address, _, dat, src_bus in can_recv:
      if src_bus == 0 and address == rx_id:
        response_found = True
        desc = "Unknown"
        if tx_id == 0x24B: desc = "Front Camera (Known)"
        elif tx_id == 0x241: desc = "EBCM (Brake)"
        elif tx_id == 0x242: desc = "PSCM (Steering EPS) - CANDIDATE!"
        elif tx_id == 0x7E0: desc = "ECM (Engine)"
        
        print(f"[FOUND] ID: 0x{tx_id:X} (Response: 0x{rx_id:X}) -> {desc}")
        print(f"        Data: {dat.hex()}")
        
        if tx_id == 0x242:
            found_eps = True
            
  print("\nScan Complete.")
  if found_eps:
      print("SUCCESS: Found 0x242! This is likely your EPS.")
      print("We can proceed with creating a Reset Tool using 0x242.")
  else:
      print("WARNING: Did not get a response from 0x242.")
      print("Try checking if the car is IGNITION ON.")

if __name__ == "__main__":
  find_eps()
