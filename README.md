# Processing scripts of 272-dimentional Motion Representation

<p align="center">
  <img src="./gif/spin_ik.gif" alt="Visualization IK" width="600"/>
</p>

> 🔄 Left: Our Representation; Right: IK failure.
> We refine the motion representation to enable directly conversion from joint rotations to SMPL body parameters, removing the need of Inverse Kinematics (IK) operation.
## 🚀 Getting Started

### 🐍 Python Virtual Environment
```sh
conda env create -f environment.yaml
conda activate mgpt
```

## 📥 Data Preparation

<details>
<summary><b> ⬇️ Download AMASS data</b></summary>

- For **HumanML3D**, **BABEL**, and **KIT-ML** dataset usage:
  - Download all "SMPL-H G" motions from the [AMASS website](https://amass.is.tue.mpg.de/download.php)
  - Place them in `datasets/amass_data`
- For **Motion-X** usage:
  - Download all `SMPL-X G`
  - Place them in `datasets/amass_data_smplx`
</details>

<details>
<summary><b>🤖 Download SMPL+H and DMPL model</b></summary>

1. Download [SMPL+H](https://mano.is.tue.mpg.de/download.php) (Extended SMPL+H model used in AMASS project)
2. Download [DMPL](https://smpl.is.tue.mpg.de/download.php) (DMPLs compatible with SMPL)
3. Place all models under `./body_model/`
</details>

<details>
<summary><b>👤 Download human model files</b></summary>

1. Download files from [Google Drive](https://drive.google.com/file/d/1y5jthVfCcMkT4cPNlyctH_AMDNz48e43/view?usp=sharing)
2. Place under `./body_model/`
</details>

<details>
<summary><b>⚙️ Process AMASS data</b></summary>

```python
python amass_process.py --index_path ./test_t2m.csv --save_dir ./output/smpl_85
```
</details>

<details>
<summary><b>📝 Generate mapping files and text files</b></summary>

Follow [UniMoCap](https://github.com/LinghaoChan/UniMoCap/tree/main?tab=readme-ov-file#2-generate-mapping-files-and-text-files) Step2 to get:
- Mapping files (.csv)
- Text files (./{dataset}_new_text)
</details>

## 🏃 Quick Start Guide

### 1. Transform SMPL to Z+ direction
```python
python face_z_transform.py --filedir ./output
```

### 2. Get global joint positions through SMPL layer
```python
python infer_get_joints.py --filedir ./output
```

### 3. Generate 272-dimensional motion representation
```python
python representation_272.py --filedir ./output
```

### 4. Calculate Mean and Std (Optional)
> We provide 272-dimentional Mean.npy and Std.npy of HumanML3D dataset under folder "mean_std/".
```python
python cal_mean_std.py --input_dir ./output/Representation_272 --output_dir ./mean_std
```

### 5. Visualize representation (Optional)
```python
python recover_visualize.py --mode rot --input_dir ./output/Representation_272 --output_dir ./visualize_result
```

### 6. Representation_272 to BVH conversion (Optional)
```python
python representation_272_to_bvh.py --gender NEUTRAL --poses ./output/Representation_272 --output ./output/Representation_272 --fps 60 --is_folder
```

## 🎬 Visualization Results

<p align="center">
  <img src="./gif/recover_rotation.gif" alt="Recover from rotation" width="45%" style="margin-right: 20px"/>
  <img src="./gif/recover_position.gif" alt="Recover from position" width="45%"/>
</p>
<p align="center">
  <em>Left: Recover from rotation &nbsp;&nbsp;&nbsp;&nbsp; Right: Recover from position</em>
</p>
