# Proof of submission: UPAR 2027 Track 1 (Codabench competition 18180)

| | |
|---|---|
| Competition | RWS2027 - UPAR Challenge Track 1: Pedestrian Attribute Recognition @ WACV2027 |
| Phase | Evaluation Phase (id 30660) |
| Account | `ali123` (participant status: approved) |
| Submission id | **1004948** |
| File | `sub03_clip_b16_calib.zip` (CLIP ViT-B/16, N1-N5) |
| Uploaded | 2026-10-20 09:31 UTC |
| Status | **Finished** |
| **Challenge_Avg** | **0.7699** |
| mA / Label_F1 | 0.7725 / 0.5596 |
| Instance Acc / Prec / Rec / F1 | 0.6512 / 0.7419 / 0.8084 / 0.7672 |
| Leaderboard position | 47 of the 50 listed entries (1st place: 0.8021) |

- `sub03_screenshot.png` (original: `sub03_leaderboard_original.jpeg`): the Codabench leaderboard rows for our team.
  Row 47 is `ali123`, submission 1004948. Row 48 is `fasihhk` (Fasih Hasan Khan), submission 1005565,
  the same sub03 package uploaded from the second team member's account, with identical scores.
- `sub03_codabench_record.json`: the submission record returned by the Codabench API.
- `sub03_ingestion_log.txt`, `sub03_scoring_log.txt`: the platform's logs for this run. They show that all
  29,248 hidden test images were predicted in 34.8 s (45 s in total, against a 300 s limit) and then scored.

## Later submissions (not scored because of the platform outage)

| Id | File | Status |
|---|---|---|
| 1006092 | `sub05_clip_b16_trainval.zip` | prediction finished on the platform (29,248 images, 46 s) but scoring never started; still "Running" |
| 1008662 | `sub07_ens_b16_l14_calib.zip` | uploaded 2026-10-21 about 13:40 UTC, queued; codabench.org stopped responding about 13:45 UTC |
