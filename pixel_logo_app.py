import streamlit as st
from PIL import Image
import io
import numpy as np
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, lab2rgb
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt

st.set_page_config(page_title="픽셀 로고 생성기", layout="centered")
st.title("🧱 픽셀 로고 생성기")
st.caption("이미지 비율을 유지하며, 색상 그룹을 정제해 깔끔한 픽셀아트를 생성하세요!")

uploaded_file = st.file_uploader("이미지를 업로드하세요 (PNG, JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼 원본 이미지", use_column_width=True)

    # 사용자 설정
    col1, col2 = st.columns(2)
    with col1:
        cols = st.slider("📐 가로 픽셀 수", min_value=9, max_value=36, value=18)
    with col2:
        num_colors = st.slider("🎨 대표 색상 그룹 수", min_value=2, max_value=6, value=4)

    col3, col4 = st.columns(2)
    with col3:
        remove_bg = st.checkbox("🧼 배경색 제거")
    with col4:
        show_grid = st.checkbox("🔲 그리드 보기")

    shape_option = st.radio("🧩 픽셀 모양 선택", ["사각형", "동그라미"])

    if remove_bg:
        bg_hex = st.color_picker("제거할 배경색 선택", "#FFFFFF")
        bg_color = tuple(int(bg_hex.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))

    # 비율 유지
    aspect_ratio = img.width / img.height
    rows = int(round(cols / aspect_ratio))

    # 이미지 축소 및 RGB → LAB
    small_img = img.resize((cols, rows), Image.NEAREST)
    small_np = np.array(small_img) / 255.0
    lab_img = rgb2lab(small_np)
    lab_flat = lab_img.reshape(-1, 3)

    # 1. KMeans로 많은 색상 추출
    kmeans = KMeans(n_clusters=12, n_init=10, random_state=42)
    labels = kmeans.fit_predict(lab_flat)
    lab_palette = kmeans.cluster_centers_

    # 2. LAB 거리 기준으로 num_colors개 대표 그룹으로 다시 군집화
    dist = distance_matrix(lab_palette, lab_palette)
    groups = []
    used = set()

    for i in range(len(lab_palette)):
        if i in used:
            continue
        group = [i]
        used.add(i)
        for j in range(i+1, len(lab_palette)):
            if j in used:
                continue
            if np.linalg.norm(lab_palette[i] - lab_palette[j]) < 15:
                group.append(j)
                used.add(j)
        groups.append(group)

    # 그룹을 대표 색으로 압축 (중심 평균)
    final_palette_lab = []
    for g in groups[:num_colors]:
        avg = np.mean([lab_palette[i] for i in g], axis=0)
        final_palette_lab.append(avg)

    final_palette_lab = np.array(final_palette_lab)

    # 3. 각 픽셀을 가장 가까운 대표색으로 매핑
    dists = distance_matrix(lab_flat, final_palette_lab)
    final_labels = np.argmin(dists, axis=1)
    mapped_lab = final_palette_lab[final_labels]

    rgb_mapped = (lab2rgb(mapped_lab.reshape(rows, cols, 3)) * 255).astype('uint8')

    clustered_img = rgb_mapped.copy()

    # 배경 제거 처리
    if remove_bg:
        bg_arr = np.array(bg_color)
        distance = np.linalg.norm(clustered_img - bg_arr, axis=2)
        mask = distance < 30
        alpha = np.where(mask, 0, 255).astype(np.uint8)
        clustered_img = np.dstack([clustered_img, alpha])
        mode = "RGBA"
    else:
        mode = "RGB"

    pixel_img = Image.fromarray(clustered_img, mode=mode)
    enlarged = pixel_img.resize((img.width, img.height), Image.NEAREST)

    # 시각화
    st.subheader("🧩 픽셀 시각화 결과")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(0, cols + 1, 1))
    ax.set_yticks(np.arange(0, rows + 1, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if show_grid:
        ax.grid(which='both', color='gray', linewidth=0.5)
    else:
        ax.grid(False)

    if shape_option == "사각형":
        for y in range(rows):
            for x in range(cols):
                color = clustered_img[y, x, :3] / 255
                rect = plt.Rectangle((x, rows - y - 1), 1, 1, color=color)
                ax.add_patch(rect)
    else:
        dot_radius = 0.48
        for y in range(rows):
            for x in range(cols):
                color = clustered_img[y, x, :3] / 255
                circle = plt.Circle((x + 0.5, rows - y - 0.5),
                                    radius=dot_radius,
                                    color=color, ec=None, linewidth=0)
                ax.add_patch(circle)

    st.pyplot(fig)

    # 다운로드
    img_byte_arr = io.BytesIO()
    enlarged.save(img_byte_arr, format='PNG')
    st.download_button(
        label="📥 픽셀 로고 다운로드 (사각형 기반)",
        data=img_byte_arr.getvalue(),
        file_name="pixel_logo.png",
        mime="image/png"
    )
