import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import io

st.set_page_config(page_title="픽셀 로고 생성기", layout="centered")
st.title("🧱 픽셀 로고 생성기")
st.caption("이미지를 픽셀 스타일로 변환하고, 색상 수와 해상도를 조절해보세요!")

uploaded_file = st.file_uploader("이미지를 업로드하세요 (PNG, JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGBA")
    st.image(img, caption="🖼 원본 이미지", use_column_width=True)

    # 슬라이더로 해상도 조절 (9x9 ~ 18x18)
    pixel_resolution = st.slider("픽셀 해상도 (최소 = 큼, 최대 = 작음)", min_value=9, max_value=18, value=12)

    # 슬라이더로 색상 수 조절
    num_colors = st.slider("로고에 사용할 색상 수", min_value=2, max_value=6, value=4)

    # 배경색 제거 여부 선택
    remove_bg = st.checkbox("배경색 제거 (흰색 계열 자동 투명 처리)")

    # 이미지 축소
    small_img = img.resize((pixel_resolution, pixel_resolution), Image.NEAREST)
    img_np = np.array(small_img)

    # 배경 제거 (흰색 계열 자동 제거)
    if remove_bg:
        bg_mask = np.all(img_np[:, :, :3] > 240, axis=-1)  # 거의 흰색인 영역
        img_np[bg_mask] = [0, 0, 0, 0]  # 투명 처리

    # KMeans로 색상 단순화
    pixels = img_np.reshape(-1, 4)
    mask = pixels[:, 3] > 0  # 투명한 픽셀은 제외
    pixels_rgb = pixels[mask][:, :3]

    kmeans = KMeans(n_clusters=num_colors, n_init=10, random_state=42)
    labels = kmeans.fit_predict(pixels_rgb)
    new_colors = kmeans.cluster_centers_.astype('uint8')

    # 색상 적용
    pixels[mask][:, :3] = new_colors[labels]
    clustered_img = pixels.reshape((pixel_resolution, pixel_resolution, 4)).astype('uint8')

    # 확대해서 출력
    output_img = Image.fromarray(clustered_img).resize((img.width, img.height), Image.NEAREST)
    st.image(output_img, caption=f"🎨 결과: {pixel_resolution}x{pixel_resolution}, {num_colors}색", use_column_width=True)

    # 다운로드
    img_byte_arr = io.BytesIO()
    output_img.save(img_byte_arr, format='PNG')
    st.download_button(
        label="📥 로고 다운로드 (투명 PNG)",
        data=img_byte_arr.getvalue(),
        file_name="pixel_logo.png",
        mime="image/png"
    )
