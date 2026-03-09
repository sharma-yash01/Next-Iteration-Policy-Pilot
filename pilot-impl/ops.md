# Pilot ops — run, storage, diagnostics

## Upload & run on EC2

**Local (from repo root):**
```bash
cd "/Users/yashaswisharma/Desktop/Academia Career/usc-gradschool/Research/self-repair-termination"
tar --exclude='pilot-impl/pilot-impl-env' --exclude='pilot-impl/__pycache__' --exclude='pilot-impl/data' -czvf pilot.tar.gz pilot-impl
scp -i /path/to/your-key.pem pilot.tar.gz ubuntu@<PUBLIC_DNS_OR_IP>:~
```

**On host (Ubuntu; user is `ubuntu`, not `ec2-user`):**
```bash
tar -xzvf ~/pilot.tar.gz -C ~
cd ~/pilot-impl
sudo apt update && sudo apt install -y python3.12-venv
python3 -m venv pilot-impl-venv && source pilot-impl-venv/bin/activate
pip install -r requirements.txt
export OPENROUTER_API_KEY="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"
nohup python run_pilot.py > pilot.log 2>&1 &
# Or in tmux: tmux new -s pilot → run the above (no nohup), then Ctrl+B D to detach
```

**Pull results to local:**
```bash
scp -i /path/to/key.pem -r ubuntu@<HOST>:~/pilot-impl/data .
```

---

## Storage

**Disk (volume):**
```bash
df -h /
lsblk
```
EBS resize in AWS console → then on host (if partition not grown): `sudo growpart /dev/nvme0n1 1` then `sudo resize2fs /dev/nvme0n1p1`. Reboot if `lsblk` still shows old size; usually visible in ~30s.

**RAM (memory):**
```bash
ps aux --sort=-%mem | head -20
# or
ps -eo pid,%mem,rss,comm --sort=-rss | head -20
```
Interactive: `top -o %MEM` or `htop` (F6 → MEM%).

**What’s using disk (by directory):**
```bash
sudo du -sh /home/ubuntu/* /home/ubuntu/.cache/* /snap /var/cache/apt 2>/dev/null | sort -rh | head -15
```

---

## Pilot diagnostics

**Is it running?**
```bash
ps aux | grep run_pilot
```

**Progress & ETA (tqdm in log):**
```bash
tail -20 ~/pilot-impl/pilot.log
```
Look for e.g. `Repair loops:  12%|...| 10/80 [02:15:00<15:30:00, 810.00s/it]` — ETA after `\[`.

**Finished problem count:**
```bash
ls ~/pilot-impl/data/trajectories/*.jsonl 2>/dev/null | wc -l
```
ETA ≈ (80 − count) × ~843 seconds.

---

## Free disk (optional)

```bash
rm -rf ~/.cache/huggingface   # re-downloads dataset next run
rm -rf ~/.cache/pip
sudo apt clean
find ~/pilot-impl -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
```

---

## Reference

- **Rough runtime:** ~843 s/problem → 80 problems ≈ 18–19 h.
- **SSH:** Use the instance **public DNS** or **public IP** (not private hostname). Ubuntu default user: `ubuntu`.
- **Disk vs RAM:** `df` = disk (files). `ps`/`top` = RAM (processes). They are independent.
