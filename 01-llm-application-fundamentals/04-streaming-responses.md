# Streaming Responses in LangChain

## مقدمه

در ساخت اپلیکیشن‌های مبتنی بر مدل‌های زبانی، فقط کیفیت پاسخ مهم نیست؛ **نحوه رسیدن پاسخ به کاربر** هم به همان اندازه مهم است. اگر مدل پاسخ خوبی بدهد اما کاربر چند ثانیه با یک صفحه ساکت یا spinner خالی روبه‌رو بماند، تجربه کاربری ضعیف می‌شود. به همین دلیل streaming یکی از مهم‌ترین قابلیت‌ها در طراحی LLM applicationها است: به جای اینکه منتظر بمانیم کل پاسخ آماده شود، خروجی را به‌صورت تدریجی و همزمان با تولید مدل به کاربر نشان می‌دهیم.

در این درس هدف فقط یاد گرفتن یک تکه کد نیست. هدف این است که بفهمیم streaming از نظر مفهومی چیست، در پایتون چگونه با generator و `yield` پیاده می‌شود، در LangChain چه تفاوتی میان `invoke()` و `stream()` وجود دارد، و در نهایت چطور این جریان را به یک UI متصل کنیم تا کاربر واقعاً پاسخ را به‌صورت زنده ببیند. اگر این منطق درست فهمیده شود، همین الگو را می‌توان در Gradio، FastAPI، WebSocket و تقریباً هر رابط تعاملی دیگری هم به کار برد.

## Context

Before this lesson, the application could already accept a user message, send the conversation history to the model, and return a complete answer. That means the chatbot was functional, but the interaction still felt batch-oriented: the user asked something, then waited, then received the full response at once.

This lesson adds the missing interaction layer: live token delivery. The project used in the lesson is a diet planning chatbot that stores conversation history and generates personalized meal-plan responses. The goal here is not to redesign the bot, but to change how the response reaches the user.

## Why LLMs Feel Slow

Large language models generate text incrementally, token by token, not all at once. Even when the final response appears as a complete paragraph, the model has actually been producing that output piece by piece behind the scenes [file:22].

This creates an important distinction between actual latency and perceived latency. Actual latency is the real time the model needs to finish generating. Perceived latency is what the user feels while waiting. Streaming improves the second one dramatically: users start seeing progress immediately instead of staring at a blank interface [file:22].

### Without streaming

- The user sends a message.
- The application waits for the full response.
- Nothing visible happens for a few seconds.
- The final answer appears all at once.

### With streaming

- The user sends a message.
- The model starts producing chunks.
- The UI updates as chunks arrive.
- The user experiences immediate feedback and visible progress [file:22].

That is why streaming matters. In many LLM products, the perceived quality of the system depends as much on responsiveness as on answer quality.

## Python Foundation: `yield` and Generators

To understand streaming correctly, the first concept to master is the difference between a regular function and a generator.

### Regular function with `return`

A normal function runs until completion and returns once.

```python
def get_full_response():
    result = call_llm()
    return result
```

Once `return` is executed, the function ends completely. Its local state is gone, and control returns to the caller.

### Generator with `yield`

A generator works differently.

```python
def stream_response():
    for chunk in call_llm_streaming():
        yield chunk
```

When Python reaches `yield`, it returns a value to the caller **without destroying the function state**. The function is paused, not finished. The next time the caller asks for another value, execution resumes from the exact point where it stopped [file:21][file:22].

### Why keeping state matters

A common confusion is this: “If everything eventually gets processed anyway, why does preserving state matter?” The answer is timing.

Without preserved state, the program must finish processing the whole response before giving anything back. With preserved state, it can process one chunk, give it to the caller, pause, and continue later. This is what makes real-time output possible [file:22].

### Function vs Generator

| Feature | Function | Generator |
|---|---|---|
| Output style | Single final value | Multiple incremental values |
| Keyword | `return` | `yield` |
| State preserved | No | Yes |
| Memory behavior | Often full result first | Incremental consumption |
| Best for | Batch output | Streaming / lazy evaluation |

### Simple simulation

The lesson uses a simplified example to demonstrate the pattern.

```python
def stream_diet_response():
    """Stream a response chunk by chunk"""
    response = "I'll help you create a healthy meal plan for your goals."

    for word in response.split():
        yield word + " "


for i, chunk in enumerate(stream_diet_response()):
    print(f"Chunk {i + 1}: {chunk.strip()}")
```

This example is intentionally simple. It does not represent a real LLM API. It only demonstrates the control-flow pattern: produce one piece, pause, resume, produce the next [file:22].

### Important correction

The chunk yielded by a real LLM stream is **not necessarily one word**. That is one of the weak points in many beginner explanations. In practice, a chunk may be:

- a single token,
- part of a word,
- a full word,
- punctuation,
- or multiple tokens grouped together [file:22].

So when the lesson says `yield chunk.text`, that does **not** mean “return each word.” It means “return each text fragment that the provider sends.”

### Type hint: `Generator[str, None, None]`

The code uses this signature:

```python
def stream_response(self, user_message: str) -> Generator[str, None, None]:
```

This follows the pattern `Generator[YieldType, SendType, ReturnType]`.

| Position | Meaning | In this lesson |
|---|---|---|
| First | Type yielded out | `str` |
| Second | Type sent in with `.send()` | `None` |
| Third | Final returned value | `None` |

So `Generator[str, None, None]` means: this generator yields strings, does not expect values to be sent into it, and does not return a final value after completion [file:22].

## Streaming in LangChain

Once the Python generator idea is clear, the LangChain part becomes much easier.

### `invoke()` vs `stream()`

A non-streaming version of an LLM call usually looks like this:

```python
response = self.model.invoke(messages)
```

That waits until the model finishes and then returns the complete response.

A streaming version looks like this:

```python
for chunk in self.model.stream(messages):
    ...
```

Here, the connection remains active and LangChain yields chunk objects as they arrive from the model provider [file:22].

### What is `chunk.text`?

In the lesson code, each streamed object is inspected with `chunk.text`. That extracts the text content of the current chunk [file:22].

This is a subtle but important point: the chunk object is not just raw text. It is usually a structured message fragment provided by LangChain. The lesson simplifies this to `chunk.text`, which is fine pedagogically, but the deeper point is that LangChain is wrapping provider events into a more convenient abstraction.

### What you can and cannot control

You do **not** directly control:

- the exact chunk size,
- the exact moment each chunk arrives,
- whether a word is split across chunk boundaries [file:22].

You **do** control:

- whether you use `invoke()` or `stream()`,
- how you accumulate chunks,
- how frequently you refresh the UI,
- whether you buffer multiple chunks before showing them.

That distinction matters because many beginners assume streaming means “one word at a time.” It does not. It means “consume partial output incrementally.”

## Implementing Streaming in `DietChatBot`

The lesson’s main logic change is the introduction of a generator-based method called `stream_response`.

```python
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
```

### Line-by-line logic

1. The method strips whitespace and exits if the message is empty.
2. It immediately stores the user message in conversation history.
3. It initializes an empty `response_content` string.
4. It iterates over `self.model.stream(...)` instead of calling `invoke()`.
5. For each non-empty chunk, it appends the text to the accumulated full response.
6. It yields the same text immediately to the caller.
7. After streaming completes, it stores the full assistant message in history [file:22].

### Why save history after streaming finishes?

This is one of the most important implementation choices in the lesson. The assistant response is stored only after the full stream finishes, not after each partial chunk [file:22].

That decision is sensible for a beginner implementation, because the database should contain a coherent final assistant message, not dozens of tiny fragments. If the app stored every chunk independently, history would become noisy and hard to reconstruct.

### Critical review

The lesson’s logic is good for teaching, but it hides a production concern: what happens if streaming fails halfway through? In the current version, partial content may be lost because the final `AIMessage` is only added after successful completion. For learning purposes this is acceptable, but for robust systems an error-handling strategy is needed.

## Connecting the Generator to the UI

Streaming is only useful if the UI consumes the generator correctly. That happens in `stream_message_handler`.

```python
def stream_message_handler(
    self, user_message: str, history: list
) -> Generator[Tuple[str, list], None, None]:
    """Send message to chatbot and stream the response."""
    if not (user_message := user_message.strip()):
        yield "", history
        return

    history.append(gr.ChatMessage(content=user_message, role="user"))
    history.append(gr.ChatMessage(content="", role="assistant"))

    yield "", history

    full_response = ""
    for chunk in self.diet_chatbot.stream_response(user_message):
        full_response += chunk
        history[-1] = gr.ChatMessage(content=full_response, role="assistant")
        yield "", history
```

### Why this works

The UI creates an empty assistant placeholder first. Then, every time a new chunk arrives, it updates the last assistant message with a slightly longer string and yields again so the frontend refreshes [file:22].

That means the same generator idea appears at two levels:

- backend generator: yields model chunks,
- UI generator: yields updated interface states.

This is the part many learners miss. Streaming is not just a model feature. It is a coordination pattern between model output and UI refresh.

### Step-by-step flow

1. The user sends a message.
2. The UI appends the user bubble.
3. The UI appends an empty assistant bubble.
4. The UI yields once so the empty assistant placeholder becomes visible.
5. The backend starts streaming chunks.
6. Each chunk is appended to `full_response`.
7. The last assistant message is replaced with the updated text.
8. The UI yields again, causing another refresh [file:22].

## End-to-End Flow

```text
User submits message
    ↓
UI handler receives input
    ↓
UI adds user message + empty assistant placeholder
    ↓
UI yields updated state
    ↓
Backend calls model.stream(history)
    ↓
Model sends chunks incrementally
    ↓
Backend yields each chunk
    ↓
UI appends chunk to full_response
    ↓
UI replaces assistant placeholder content
    ↓
UI yields refreshed history
    ↓
User sees the response grow in real time
```

## Final Checklist

A correct mental model for this lesson should include all of these points:

- An LLM generates text incrementally, not as one giant string [file:22].
- `yield` pauses a function and preserves its state [file:21][file:22].
- A generator is what makes incremental output possible in Python [file:22].
- `stream()` is the streaming counterpart of `invoke()` in LangChain [file:22].
- `chunk.text` is a text fragment, not necessarily a complete word [file:22].
- The backend streams chunks, but the UI must also be designed to refresh incrementally [file:22].
- Accumulating partial chunks into one final response is useful for storing clean conversation history [file:22].

## Summary

This lesson teaches more than a convenience feature. It teaches a core design pattern for LLM applications: incremental generation, incremental transport, and incremental rendering.

The real insight is this: streaming is not one trick and not one API call. It is a chain of compatible behaviors across Python control flow, LangChain model access, and UI update logic. Once that is understood, the same idea can be reused in many other frameworks and architectures [file:22].
