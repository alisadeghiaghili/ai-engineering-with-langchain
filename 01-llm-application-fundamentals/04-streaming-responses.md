# Streaming Responses in LangChain

> **بخش آخر از کورس:** LLM Application Fundamentals with LangChain  
> **سطح:** Intermediate  
> **پیش‌نیاز:** آشنایی با Python generators، LangChain basics، و ساختار `DietChatBot` از دروس قبل

---

## چرا Streaming؟

LLMها معمولاً چند ثانیه طول می‌کشند تا یک پاسخ کامل تولید کنند. بدون streaming، کاربر باید تمام این مدت منتظر بماند — مثل اینکه ایمیل می‌نویسید و تا وقتی کل ایمیل نوشته نشده، هیچ حرفی روی صفحه نمایش داده نمی‌شود.

با streaming، هر توکن به محض تولید به کاربر نمایش داده می‌شود. این تجربه بسیار طبیعی‌تر و responsive‌تر است.

**مزایای اصلی:**
- کاربر فوری بازخورد می‌گیرد
- **Perceived latency** به شدت کاهش می‌یابد (حتی اگر زمان کل یکسان باشد)
- اپلیکیشن حرفه‌ای‌تر به نظر می‌رسد
- می‌توان پردازش را موازی با دریافت داده انجام داد

---

## پایه: `yield` در مقابل `return`

قبل از پیاده‌سازی streaming، باید تفاوت بنیادی `return` و `yield` را درک کرد.

### تابع معمولی با `return`

```python
def get_full_response():
    result = call_llm()  # صبر می‌کند تا همه چیز آماده شود
    return result        # یک بار خروجی می‌دهد و تمام
```

- تابع اجرا می‌شود، نتیجه برمی‌گردد، function از stack حذف می‌شود
- هیچ state‌ای حفظ نمی‌شود
- تنها یک خروجی دارد

### Generator با `yield`

```python
def stream_response():
    for chunk in call_llm_streaming():
        yield chunk  # اجرا را pause می‌کند و یک مقدار برمی‌گرداند
                     # و سپس از همان نقطه ادامه می‌دهد
```

- `yield` اجرا را **pause** می‌کند، نه **stop**
- تمام local variableها و موقعیت loop حفظ می‌شوند
- هر بار که از generator مقدار خواسته شود، از نقطه pause ادامه می‌دهد

### جدول مقایسه

| ویژگی | `return` (Function) | `yield` (Generator) |
|---|---|---|
| تعداد خروجی | یک بار | چند بار |
| حفظ state | ❌ | ✅ |
| نوع بازگشتی | مقدار مستقیم | `generator` object |
| مصرف حافظه | کل داده را لود می‌کند | یک chunk در هر بار |
| `return` اجباری؟ | بله | اختیاری |

### سوال رایج: "مگر در نهایت همه چیز پردازش نمی‌شود؟"

بله، در نهایت همه چیز پردازش می‌شود — اما **تفاوت در زمان‌بندی و ترتیب** است.

بدون `yield`:
1. کل داده در RAM لود می‌شود
2. کل پردازش انجام می‌شود
3. **سپس** اولین output به کاربر نمایش داده می‌شود

با `yield`:
1. chunk اول پردازش می‌شود
2. **بلافاصله** به کاربر نمایش داده می‌شود
3. chunk دوم پردازش می‌شود
4. **بلافاصله** به کاربر نمایش داده می‌شود
5. ...

---

## مثال ساده: شبیه‌سازی Streaming

```python
def stream_diet_response():
    """Stream a response chunk by chunk"""
    response = "I'll help you create a healthy meal plan for your goals."
    
    for word in response.split():
        yield word + " "


# مصرف generator
print("LLM streaming response:\n")
for i, chunk in enumerate(stream_diet_response()):
    print(f"Chunk {i + 1}: {chunk.strip()}")
```

**خروجی:**
```
LLM streaming response:

Chunk 1: I'll
Chunk 2: help
Chunk 3: you
Chunk 4: create
Chunk 5: a
Chunk 6: healthy
Chunk 7: meal
Chunk 8: plan
Chunk 9: for
Chunk 10: your
Chunk 11: goals.
```

> **نکته مهم:** این مثال برای نمایش مکانیزم است. در واقعیت LLM، chunk‌ها لزوماً کلمه به کلمه نیستند — ممکن است token، نیمی از کلمه، یا چند کلمه باشند.

---

## Type Hint برای Generator

وقتی یک تابع Generator برمی‌گرداند، type hint به این شکل است:

```python
from typing import Generator

def stream_response(self, user_message: str) -> Generator[str, None, None]:
    ...
```

سه parameter در `Generator[YieldType, SendType, ReturnType]`:

| Parameter | توضیح | در مثال ما |
|---|---|---|
| `YieldType` | نوع مقداری که `yield` می‌دهد | `str` — متن chunk‌ها |
| `SendType` | نوع مقداری که با `.send()` ارسال می‌شود | `None` — استفاده نمی‌کنیم |
| `ReturnType` | نوع مقداری که بعد از پایان `return` می‌دهد | `None` — چیزی return نمی‌کنیم |

برای اکثر موارد streaming، `Generator[str, None, None]` کافی است.

---

## پیاده‌سازی واقعی: LangChain + `model.stream()`

### Logic Layer — `stream_response` در `DietChatBot`

```python
from typing import Generator
from langchain_core.messages import HumanMessage, AIMessage

def stream_response(self, user_message: str) -> Generator[str, None, None]:
    """Stream AI response chunks for a user message."""
    # اگر پیام خالی بود، چیزی yield نمی‌کنیم
    if not (user_message := user_message.strip()):
        return

    # پیام کاربر را در تاریخچه ذخیره می‌کنیم
    self.history.add_message(HumanMessage(content=user_message))

    # Streaming response
    response_content = ""
    for chunk in self.model.stream(self.history.get_messages()):
        if chunk.text:
            response_content += chunk.text  # برای ذخیره کامل پاسخ
            yield chunk.text                # برای نمایش فوری

    # پاسخ کامل را در تاریخچه ذخیره می‌کنیم
    self.history.add_message(AIMessage(content=response_content))
```

**نکات کلیدی:**

- `self.model.stream()` به جای `self.model.invoke()` — connection باز می‌ماند و chunk‌ها به تدریج می‌آیند
- `chunk.text` — متن خام هر chunk (بعضی chunk‌ها ممکن است خالی باشند، پس `if chunk.text` داریم)
- `response_content += chunk.text` — تجمیع برای ذخیره در history
- `yield chunk.text` — ارسال فوری به caller
- ذخیره در history **بعد از** اتمام streaming انجام می‌شود — نه در حین آن

**چرا history را در طول streaming آپدیت نمی‌کنیم؟**  
چون تا قبل از تکمیل پاسخ، محتوای آن ناقص است. اگر سشن قطع شود، پاسخ نیمه‌کاره را ذخیره نمی‌خواهیم.

### UI Layer — `stream_message_handler` در `PersistentChatBotUI`

```python
from typing import Generator, Tuple
import gradio as gr

def stream_message_handler(
    self, user_message: str, history: list
) -> Generator[Tuple[str, list], None, None]:
    """Send message to chatbot and stream the response."""
    if not (user_message := user_message.strip()):
        yield "", history
        return

    # پیام کاربر را به UI اضافه می‌کنیم
    history.append(gr.ChatMessage(content=user_message, role="user"))
    # یک placeholder خالی برای پاسخ assistant
    history.append(gr.ChatMessage(content="", role="assistant"))

    # UI را با پیام کاربر و placeholder خالی آپدیت می‌کنیم
    yield "", history

    # Streaming
    full_response = ""
    for chunk in self.diet_chatbot.stream_response(user_message):
        full_response += chunk
        # آخرین پیام (placeholder) را با محتوای فعلی آپدیت می‌کنیم
        history[-1] = gr.ChatMessage(content=full_response, role="assistant")
        yield "", history  # UI refresh می‌شود
```

**جریان اجرا قدم به قدم:**

1. کاربر پیام می‌فرستد
2. پیام کاربر و یک bubble خالی برای assistant به UI اضافه می‌شود
3. `yield "", history` — UI آپدیت می‌شود، کاربر پیام خود را می‌بیند
4. Loop شروع می‌شود — هر chunk از `stream_response` دریافت می‌شود
5. chunk به `full_response` اضافه می‌شود
6. آخرین message در history آپدیت می‌شود
7. `yield "", history` — UI دوباره رفرش می‌شود، کاربر رشد پاسخ را می‌بیند
8. تا پایان loop این تکرار می‌شود

---

## معماری کامل سیستم

```
User Input
    │
    ▼
stream_message_handler (UI Layer)
    │  ├── append user message to UI history
    │  ├── append empty assistant placeholder
    │  └── yield → UI updates (user sees their message)
    │
    ▼
stream_response (Logic Layer)
    │  ├── add HumanMessage to DB history
    │  └── model.stream(messages) ──► LangChain API ──► LLM
    │           │
    │           ▼ chunk by chunk
    │       yield chunk.text
    │
    ▼ (back in UI Layer)
    ├── full_response += chunk
    ├── history[-1] = updated assistant message
    └── yield → UI updates (user sees token appearing)
    
    [After all chunks received]
    └── add complete AIMessage to DB history
```

---

## کنترل‌هایی که دارید (و ندارید)

### آنچه **نمی‌توانید** کنترل کنید:
- اندازه chunk — توسط LLM provider تعیین می‌شود
- سرعت generation — به مدل و بار سرور بستگی دارد
- مرز chunk‌ها — ممکن است کلمه را نصف کند

### آنچه **می‌توانید** کنترل کنید:
- streaming فعال/غیرفعال (`stream()` vs `invoke()`)
- پردازش هر chunk (فیلتر، transform، buffer)
- فرکانس آپدیت UI (می‌توانید چند chunk را buffer کنید قبل از yield)
- نحوه نمایش (typing indicator، progress bar، و غیره)

---

## ساختار فایل‌های پروژه

```
diet-chatbot/
├── logic.py      # DietChatBot class با stream_response
├── ui.py         # PersistentChatBotUI با stream_message_handler  
├── main.py       # نقطه ورود
└── prompt.md     # System prompt
```

### `logic.py` — کلاس کامل `DietChatBot`

```python
import datetime
import sqlite3
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_community.chat_message_histories import SQLChatMessageHistory
from dotenv import load_dotenv
from typing import List, Optional, Generator

load_dotenv()

DEFAULT_MODEL = "gpt-4o-mini"
DB_CONNECTION_STRING = "sqlite:///conversations.db"


class DietChatBot:
    def __init__(self, session_id: Optional[str] = None):
        with open("prompt.md", "r") as f:
            system_message = f.read()

        self.model = init_chat_model(model=DEFAULT_MODEL, temperature=0)
        self.system_msg = SystemMessage(content=system_message)
        self._initialize_session(session_id)
        self._initialize_history()

    def _initialize_session(self, session_id: Optional[str] = None) -> None:
        self.session_id = self._generate_session_id() if not session_id else session_id

    def _generate_session_id(self) -> str:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"Diet Chat - {timestamp}"

    def _initialize_history(self) -> None:
        self.history = SQLChatMessageHistory(
            session_id=self.session_id, connection_string=DB_CONNECTION_STRING
        )
        if not self.history.get_messages():
            self.history.add_message(self.system_msg)

    def stream_response(self, user_message: str) -> Generator[str, None, None]:
        """Stream AI response chunks for a user message."""
        if not (user_message := user_message.strip()):
            return

        self.history.add_message(HumanMessage(content=user_message))

        response_content = ""
        for chunk in self.model.stream(self.history.get_messages()):
            if chunk.text:
                response_content += chunk.text
                yield chunk.text

        self.history.add_message(AIMessage(content=response_content))

    def new_session(self) -> None:
        self._initialize_session()
        self._initialize_history()

    def load_session(self, session_id: str) -> None:
        if not session_id:
            return
        self._initialize_session(session_id)
        self._initialize_history()

    def get_messages(self) -> List[BaseMessage]:
        return self.history.get_messages()

    @staticmethod
    def get_previous_conversations() -> List[str]:
        query = """
        SELECT session_id FROM message_store 
        GROUP BY session_id 
        HAVING COUNT(*) > 1
        ORDER BY session_id DESC
        """
        db_path = DB_CONNECTION_STRING.replace("sqlite:///", "")
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
        cursor.execute(query)
        return [row[0] for row in cursor.fetchall()]
```

---

## نکات پیشرفته (فراتر از کورس)

### Async Streaming

برای production، `astream()` به جای `stream()` کارایی بهتری دارد:

```python
async def astream_response(self, user_message: str):
    """Async version for better concurrency."""
    self.history.add_message(HumanMessage(content=user_message))
    
    response_content = ""
    async for chunk in self.model.astream(self.history.get_messages()):
        if chunk.text:
            response_content += chunk.text
            yield chunk.text
    
    self.history.add_message(AIMessage(content=response_content))
```

### Error Handling در Streaming

اگر connection در حین streaming قطع شود، باید آن را handle کنید:

```python
def stream_response(self, user_message: str) -> Generator[str, None, None]:
    self.history.add_message(HumanMessage(content=user_message))
    
    response_content = ""
    try:
        for chunk in self.model.stream(self.history.get_messages()):
            if chunk.text:
                response_content += chunk.text
                yield chunk.text
    except Exception as e:
        yield f"\n[Error: {str(e)}]"
    finally:
        # حتی در صورت خطا، آنچه داریم را ذخیره می‌کنیم
        if response_content:
            self.history.add_message(AIMessage(content=response_content))
```

### Chunk Buffering (بهینه‌سازی UI)

اگر UI updates خیلی مکرر هستند و performance را تحت تأثیر می‌گذارند:

```python
def stream_message_handler(self, user_message, history):
    # ...
    full_response = ""
    buffer = ""
    BUFFER_SIZE = 5  # هر 5 chunk یک UI update
    
    for i, chunk in enumerate(self.diet_chatbot.stream_response(user_message)):
        full_response += chunk
        buffer += chunk
        
        if len(buffer) >= BUFFER_SIZE:
            history[-1] = gr.ChatMessage(content=full_response, role="assistant")
            yield "", history
            buffer = ""
    
    # آخرین buffer را flush می‌کنیم
    if buffer:
        history[-1] = gr.ChatMessage(content=full_response, role="assistant")
        yield "", history
```

> **هشدار:** buffering ممکن است streaming experience را کمتر smooth کند. فقط در صورت مشکل performance استفاده کنید.

---

## خلاصه

| مفهوم | نکته کلیدی |
|---|---|
| `yield` vs `return` | `yield` اجرا را pause می‌کند و state را حفظ می‌کند |
| `Generator[str, None, None]` | yield type, send type, return type |
| `model.stream()` | connection را باز نگه می‌دارد و chunk به chunk yield می‌کند |
| `model.invoke()` | منتظر پاسخ کامل می‌ماند |
| UI pattern | placeholder خالی + آپدیت در هر chunk + yield برای refresh |
| History | بعد از اتمام streaming ذخیره می‌شود، نه در حین آن |
| Chunk size | توسط LLM provider تعیین می‌شود، قابل کنترل مستقیم نیست |

---

*این جزوه بخشی از مجموعه [AI Engineering with LangChain](https://github.com/alisadeghiaghili/ai-engineering-with-langchain) است.*
