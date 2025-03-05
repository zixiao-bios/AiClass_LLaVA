import requests
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

# 本地模型路径（需替换为实际路径）
# model_path = "/mnt/workspace/llava-1.5-7b-hf"
model_path = "/mnt/workspace/llava-interleave-qwen-0.5b-hf"
lora_path = "/mnt/workspace/lora_llava_finetuned"

# 配置量化参数
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,            # INT8 量化
    llm_int8_skip_modules=['vision_tower', 'multi_modal_projector'],  # 跳过视觉编码器量化
    llm_int8_threshold=6.0,        # 激活值异常阈值
)

# 加载模型参数说明：
# pretrained_model_name_or_path: 模型本地路径
# torch_dtype=torch.float16: 半精度浮点节省50%显存
# low_cpu_mem_usage=True: 分片加载减少内存峰值
# quantization_config=bnb_config: 指定量化配置
model = LlavaForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    # quantization_config=bnb_config,
)

if lora_path:
    model = PeftModel.from_pretrained(model, lora_path)

# 使用量化时，模型会自动传到 GPU，不支持手动指定
model = model.to(0)
model.eval()

# 加载处理器参数说明：
# 与模型同路径确保tokenizer和image_processor版本匹配
processor = AutoProcessor.from_pretrained(model_path)

# 下载图片参数说明：
# stream=True: 流式传输避免大文件内存溢出
img_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
response = requests.get(img_url, stream=True)
raw_image = Image.open(response.raw)


while True:
    user_input = input("🤗：")
    if user_input == "exit":
        break

    if user_input == "image":
        img_url = input("图片链接：")
        response = requests.get(img_url, stream=True)
        raw_image = Image.open(response.raw)
        continue

    conversation = [{
        "role": "user",
        "content": [
            {"type": "text", "text": user_input},
            {"type": "image"},
        ],
    }]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(
        images=raw_image,
        text=prompt,
        return_tensors='pt'
    ).to(0)
    output = model.generate(
        **inputs,
        max_new_tokens=200,
    )
    output_text = processor.decode(output[0], skip_special_tokens=True)
    # 只输出 ‘ASSISTANT’ 后面的文本
    print(f'🤖：{output_text.split("ASSISTANT:")[-1].strip()}')
