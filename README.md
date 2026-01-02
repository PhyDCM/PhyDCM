# PhyDCM v3.0.0 — Release Notes (Performance & Packaging Modernization)

PhyDCM v3.0.0 focuses on **import-time performance**, **optional heavy dependencies**, and **robust model loading diagnostics** while preserving the library’s original research-oriented identity.

---

## Visual Identity (Logo + Team)

<p align="center">
  <img src="https://raw.githubusercontent.com/PhyDCM/PhyDCM/main/assets/logo.jpg" alt="PhyDCM Logo" width="160"/>
</p>

### Research Team

<table>
  <tr>
    <td align="center" width="220">
      <b>Supervisor</b><br/>
      <img src="https://raw.githubusercontent.com/PhyDCM/PhyDCM/main/assets/team/dr_hayder.jpg" alt="Dr. Hayder" width="160"/><br/>
      <sub><b>Dr. Hayder</b></sub>
    </td>
    <td align="center" width="220">
      <b>Students</b><br/>
      <img src="https://raw.githubusercontent.com/PhyDCM/PhyDCM/main/assets/team/mohammed_hadi.jpg" alt="Mohammed Hadi" width="120"/><br/>
      <sub><b>Mohammed Hadi</b></sub>
    </td>
    <td align="center" width="220">
      <b>Students</b><br/>
      <img src="https://raw.githubusercontent.com/PhyDCM/PhyDCM/main/assets/team/mohammed_hassan.jpg" alt="Mohammed Hassan" width="120"/><br/>
      <sub><b>Mohammed Hassan</b></sub>
    </td>
    <td align="center" width="220">
      <b>Students</b><br/>
      <img src="https://raw.githubusercontent.com/PhyDCM/PhyDCM/main/assets/team/haider_ali.jpg" alt="Haider Ali" width="120"/><br/>
      <sub><b>Haider Ali</b></sub>
    </td>
    <td align="center" width="220">
      <b>Students</b><br/>
      <img src="https://raw.githubusercontent.com/PhyDCM/PhyDCM/main/assets/team/ali_hussein.jpg" alt="Ali Hussein" width="120"/><br/>
      <sub><b>Ali Hussein</b></sub>
    </td>
  </tr>
</table>

> Add these files to your repository:
> - `assets/logo.jpg`
> - `assets/team/dr_hayder.jpg`
> - `assets/team/mohammed_hadi.jpg`
> - `assets/team/mohammed_hassan.jpg`
> - `assets/team/haider_ali.jpg`
> - `assets/team/ali_hussein.jpg`

---

## What Changed in v3.0.0

| Area | v2.4.0 Behavior | v3.0.0 Improvement | Impact |
|------|------------------|-------------------|--------|
| Import time | Heavy imports could slow down `import phydcm` | **Lazy imports** for optional components | Faster startup |
| ML runtime | TensorFlow imported/required early | TensorFlow becomes **optional** (`phydcm[ml]`) | Prevents DLL issues during import |
| Image IO | OpenCV required by default | OpenCV becomes **optional** (`phydcm[cv]`) | Smaller baseline install |
| GUI | PyQt6 required by default | GUI becomes **optional** (`phydcm[ui]`) | CLI/SDK installs stay light |
| Model loading | Silent or unclear failures | **Colored terminal diagnostics** (green/yellow/red) | Faster troubleshooting |
| Packaging | Mixed metadata sources | Modern packaging via **pyproject.toml** | Cleaner publishing |

---

## Terminal Diagnostics (Colored Status)

PhyDCM prints concise, research-grade status messages:

- **Green**: model loaded successfully  
- **Yellow**: non-fatal warning (e.g., labels missing)  
- **Red**: model missing / failed to load  

---

## Installation

### Minimal (lightweight)
```bash
pip install phydcm==3.0.0
