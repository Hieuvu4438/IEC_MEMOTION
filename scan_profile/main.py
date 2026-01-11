import os
import logging
import json
from typing import Dict, Any, Optional, Union
from pathlib import Path

from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image, UnidentifiedImageError


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("GeminiMedCore")

# Tải biến môi trường
load_dotenv()

class GeminiMedicalAssistant:
    """
    Trợ lý Y tế sử dụng Google Gemini API để xử lý hình ảnh đơn thuốc và nhận diện thuốc.
    Thiết kế tối ưu cho dự án SeniorCare (Lão khoa).
    """

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """Khởi tạo kết nối với Gemini API."""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.critical("Không tìm thấy GOOGLE_API_KEY trong biến môi trường .env")
            raise ValueError("GOOGLE_API_KEY là bắt buộc.")

        genai.configure(api_key=api_key)
        
        # Sử dụng model Flash cho tốc độ và hiệu quả chi phí cao nhất
        self.model = genai.GenerativeModel(model_name)
        logger.info(f"Đã khởi tạo Gemini Assistant với model: {model_name}")

        # Cấu hình sinh nội dung: Nhiệt độ thấp để đảm bảo tính chính xác, ép buộc trả về JSON
        self.generation_config = genai.types.GenerationConfig(
            temperature=0.1, 
            response_mime_type="application/json"
        )

    def _load_image(self, image_source: Union[str, Path]) -> Optional[Image.Image]:
        """Helper function: Tải và kiểm tra hình ảnh an toàn."""
        try:
            img_path = Path(image_source)
            if not img_path.exists():
                logger.error(f"File ảnh không tồn tại: {img_path}")
                return None
            
            img = Image.open(img_path)
            logger.debug(f"Đã tải ảnh thành công: {img_path.name}")
            return img
        except UnidentifiedImageError:
            logger.error(f"File không phải là định dạng ảnh hợp lệ: {image_source}")
            return None
        except Exception as e:
            logger.error(f"Lỗi không xác định khi tải ảnh {image_source}: {e}")
            return None

    def _safe_generate(self, prompt_parts: list) -> Dict[str, Any]:
        """Helper function: Gọi API và xử lý lỗi tập trung."""
        try:
            response = self.model.generate_content(
                prompt_parts,
                generation_config=self.generation_config
            )
            # Parse JSON từ phản hồi của Gemini
            return json.loads(response.text)
        except json.JSONDecodeError:
            logger.error("Gemini phản hồi không phải là JSON hợp lệ.")
            # Trong môi trường production, có thể lưu lại response.text để debug
            return {"error": "Invalid JSON response from AI Model", "raw_response": response.text}
        except Exception as e:
            logger.error(f"Lỗi khi gọi Gemini API: {e}")
            return {"error": str(e)}

    # ================== NHÁNH 1: NHẬN DIỆN THUỐC ==================
    def identify_pill_from_image(self, image_path: str) -> Dict[str, Any]:
        """
        Nhánh 1: Xác định loại thuốc từ hình ảnh viên thuốc/vỉ thuốc/hộp thuốc.
        """
        img = self._load_image(image_path)
        if not img:
            return {"error": "Image loading failed"}

        logger.info(f"Bắt đầu nhận diện thuốc từ ảnh: {Path(image_path).name}")

        
        prompt = """
        Bạn là một dược sĩ AI chuyên nghiệp. Hãy phân tích hình ảnh này và xác định loại thuốc.
        Trả về kết quả dưới dạng JSON với các trường sau. Nếu không chắc chắn, hãy để giá trị là null hoặc "Không rõ".
        Yêu cầu sử dụng tiếng Việt.

        JSON Schema mong muốn:
        {
          "ten_thuoc_thuong_mai": "Tên biệt dược (ví dụ: Panadol Extra)",
          "ten_hoat_chat": ["Danh sách hoạt chất chính (ví dụ: Paracetamol, Caffeine)"],
          "ham_luong": "Ví dụ: 500mg",
          "dang_bao_che": "Ví dụ: Viên nén bao phim, viên nang",
          "cong_dung_chinh": "Tóm tắt ngắn gọn công dụng chính (phù hợp cho người cao tuổi đọc)",
          "luu_y_quan_trong": "Cảnh báo quan trọng nhất nếu có (ví dụ: Có thể gây buồn ngủ)",
          "do_tin_cay": "Mức độ tự tin của bạn về nhận định này (Cao/Trung bình/Thấp)"
        }
        """
        
        return self._safe_generate([prompt, img])

    # ================== NHÁNH 2: ĐỌC HỒ SƠ/ĐƠN THUỐC ==================
    def process_medical_document(self, image_path: str) -> Dict[str, Any]:

        img = self._load_image(image_path)
        if not img:
            return {"error": "Image loading failed"}

        logger.info(f"Bắt đầu xử lý tài liệu y tế: {Path(image_path).name}")

        prompt = """
        Bạn là một trợ lý y khoa AI chuyên về lão khoa. Nhiệm vụ của bạn là trích xuất thông tin từ hình ảnh đơn thuốc hoặc hồ sơ bệnh án này.
        Hãy xử lý cẩn thận cả chữ in và đặc biệt là chữ viết tay (có thể rất khó đọc).

        Yêu cầu bắt buộc:
        1. Trả về kết quả CHỈ ở định dạng JSON.
        2. Sử dụng Tiếng Việt.
        3. Nếu một trường thông tin không có trong ảnh, hãy để giá trị là null (không được bịa đặt).

        Đây là cấu trúc JSON mẫu bạn PHẢI tuân theo:
        {
          "thong_tin_benh_nhan": {
            "ho_ten": "Họ và tên đầy đủ",
            "tuoi_hoac_nam_sinh": "Ví dụ: 75 tuổi hoặc 1949",
            "gioi_tinh": "Nam/Nữ"
          },
          "thong_tin_kham": {
            "ngay_kham": "DD/MM/YYYY",
            "noi_kham": "Tên bệnh viện/phòng khám",
            "bac_si_ke_don": "Tên bác sĩ"
          },
          "chan_doan": [
            "Danh sách các bệnh được chẩn đoán (ví dụ: Tăng huyết áp độ 2, Đái tháo đường type 2)"
          ],
          "don_thuoc": [
            {
              "stt": 1,
              "ten_thuoc_va_ham_luong": "Ví dụ: Amlodipin 5mg",
              "so_luong": "Tổng số lượng cấp (ví dụ: 30 viên)",
              "cach_dung_chi_tiet": "Ví dụ: Uống 1 viên vào buổi sáng sau ăn",
              "ghi_chu_cua_bac_si": "Lưu ý đặc biệt cho thuốc này (nếu có)"
            }
            // Thêm các thuốc tiếp theo vào mảng này
          ],
          "loi_dan_chung": "Các lời dặn dò khác của bác sĩ về chế độ ăn uống, tái khám..."
        }
        """

        return self._safe_generate([prompt, img])


if __name__ == "__main__":
    try:
        assistant = GeminiMedicalAssistant()
        
        print("-" * 30 + " TEST NHÁNH 1: NHẬN DIỆN THUỐC " + "-" * 30)
        # Thay 'path/to/your/pill_image.jpg' bằng đường dẫn ảnh thật của bạn
        pill_image_path = "test_images/vien_thuoc_mau.jpg" 
        # Tạo ảnh giả để code không lỗi nếu bạn chưa thay đường dẫn
        Path("test_images").mkdir(exist_ok=True)
        if not Path(pill_image_path).exists():
             Image.new('RGB', (60, 30), color='red').save(pill_image_path)
             logger.warning("Đã tạo ảnh giả để test. Hãy thay bằng ảnh thật!")

        pill_result = assistant.identify_pill_from_image(pill_image_path)
        print(json.dumps(pill_result, indent=2, ensure_ascii=False))


        print("\n" + "-" * 30 + " TEST NHÁNH 2: ĐỌC ĐƠN THUỐC " + "-" * 30)
        # Thay 'path/to/your/prescription.jpg' bằng đường dẫn ảnh thật
        prescription_path = "test_images/don_thuoc_mau.jpg"
        if not Path(prescription_path).exists():
             Image.new('RGB', (100, 100), color='white').save(prescription_path)
             logger.warning("Đã tạo ảnh giả để test. Hãy thay bằng ảnh thật!")

        doc_result = assistant.process_medical_document(prescription_path)
        print(json.dumps(doc_result, indent=2, ensure_ascii=False))
        
    except ValueError as ve:
        logger.critical(ve)
    except Exception as e:
        logger.critical(f"Lỗi không mong muốn trong quá trình test: {e}")