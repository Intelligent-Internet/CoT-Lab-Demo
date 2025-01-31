from openai import OpenAI
from dotenv import load_dotenv
import os
import threading
import time
import gradio as gr
from lang import LANGUAGE_CONFIG

# 环境变量预校验
load_dotenv(override=True)
required_env_vars = ["API_KEY", "API_URL", "API_MODEL"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise EnvironmentError(
        f"Missing required environment variables: {', '.join(missing_vars)}"
    )


class AppConfig:
    DEFAULT_THROUGHPUT = 10
    SYNC_THRESHOLD_DEFAULT = 0
    API_TIMEOUT = 20



class DynamicState:
    """动态UI状态"""

    def __init__(self):
        self.should_stream = False
        self.stream_completed = False
        self.in_cot = True
        self.current_language = "en"
        self.waiting_api = False  # 新增等待状态标志
        self.label_passthrough = False

    def control_button_handler(self):
        original_state = self.should_stream
        self.should_stream = not self.should_stream

        # 当从暂停->生成时激活等待状态
        if not original_state and self.should_stream:
            self.waiting_api = True
            self.stream_completed = False

        return self.ui_state_controller()

    def ui_state_controller(self):
        """生成动态UI组件状态"""
        # [control_button, thought_editor, reset_button]
        lang_data = LANGUAGE_CONFIG[self.current_language]
        control_value = (
            lang_data["pause_btn"] if self.should_stream else lang_data["generate_btn"]
        )
        control_variant = "secondary" if self.should_stream else "primary"
        # 处理等待状态显示
        if self.waiting_api and self.should_stream:
            status_suffix = lang_data["waiting_api"]
        elif self.waiting_api and not self.should_stream:
            status_suffix = lang_data["api_retry"]
        else:
            status_suffix = (
                lang_data["completed"]
                if self.stream_completed
                else lang_data["interrupted"]
            )
        editor_label = f"{lang_data['editor_label']} - {status_suffix}"
        output = (
            gr.update(value=control_value, variant=control_variant),
            gr.update() if self.label_passthrough else gr.update(label=editor_label),
            gr.update(interactive=not self.should_stream),
        )
        self.label_passthrough = False
        return output
    
    def reset_workspace(self):
        """重置工作区状态"""
        self.stream_completed = False
        self.should_stream = False
        self.in_cot = True
        self.waiting_api = False
        return self.ui_state_controller() + (
            "",
            "",
            LANGUAGE_CONFIG["en"]["bot_default"],
        )


class CoordinationManager:
    """管理人类与AI的协同节奏"""

    def __init__(self, paragraph_threshold, initial_content):
        self.paragraph_threshold = paragraph_threshold
        self.initial_paragraph_count = initial_content.count("\n\n")
        self.triggered = False

    def should_pause_for_human(self, current_content):
        if self.paragraph_threshold <= 0 or self.triggered:
            return False

        current_paragraphs = current_content.count("\n\n")
        if (
            current_paragraphs - self.initial_paragraph_count
            >= self.paragraph_threshold
        ):
            self.triggered = True
            return True
        return False


class ConvoState:
    """State of current ROUND of convo"""

    def __init__(self):
        self.throughput = AppConfig.DEFAULT_THROUGHPUT
        self.sync_threshold = AppConfig.SYNC_THRESHOLD_DEFAULT
        self.current_language = "en"
        self.convo = []
        self.initialize_new_round()
        self.is_error = False

    def initialize_new_round(self):
        self.current = {}
        self.current["user"] = ""
        self.current["cot"] = ""
        self.current["result"] = ""
        self.convo.append(self.current)

    def flatten_output(self):
        output = []
        for round in self.convo:
            output.append({"role": "user", "content": round["user"]})
            if len(round["cot"]) > 0:
                output.append(
                    {
                        "role": "assistant",
                        "content": round["cot"],
                        "metadata": {"title": f"Chain of Thought"},
                    }
                )
            if len(round["result"]) > 0:
                output.append({"role": "assistant", "content": round["result"]})
        return output

    def generate_ai_response(self, user_prompt, current_content, dynamic_state):
        lang_data = LANGUAGE_CONFIG[self.current_language]
        dynamic_state.stream_completed = False
        full_response = current_content
        api_client = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("API_URL"),
            timeout=AppConfig.API_TIMEOUT,
        )
        coordinator = CoordinationManager(self.sync_threshold, current_content)

        try:

            # 初始等待状态更新
            if dynamic_state.waiting_api:
                status = lang_data["waiting_api"]
                editor_label = f"{lang_data['editor_label']} - {status}"
                yield full_response, gr.update(
                    label=editor_label
                ), self.flatten_output()

            coordinator = CoordinationManager(self.sync_threshold, current_content)
            messages = [
                {"role": "user", "content": user_prompt},
                {
                    "role": "assistant",
                    "content": f"<think>\n{current_content}",
                    "prefix": True,
                },
            ]
            self.current["user"] = user_prompt
            response_stream = api_client.chat.completions.create(
                model=os.getenv("API_MODEL"),
                messages=messages,
                stream=True,
                timeout=AppConfig.API_TIMEOUT,
                top_p=0.95,
                temperature=0.6,
            )
            for chunk in response_stream:
                chunk_content = chunk.choices[0].delta.content
                if coordinator.should_pause_for_human(full_response) and dynamic_state.in_cot:
                    dynamic_state.should_stream = False
                if not dynamic_state.should_stream:
                    break

                if chunk_content:
                    dynamic_state.waiting_api = False
                    full_response += chunk_content
                    # Update Convo State
                    think_complete = "</think>" in full_response
                    dynamic_state.in_cot = not think_complete
                    if think_complete:
                        self.current["cot"], self.current["result"] = (
                            full_response.split("</think>")
                        )
                    else:
                        self.current["cot"], self.current["result"] = (
                            full_response,
                            "",
                        )
                    status = (
                        lang_data["loading_thinking"]
                        if dynamic_state.in_cot
                        else lang_data["loading_output"]
                    )
                    editor_label = f"{lang_data['editor_label']} - {status}"
                    yield self.current["cot"] + ("</think>" if think_complete else ""), gr.update(
                        label=editor_label
                    ), self.flatten_output()

                    interval = 1.0 / self.throughput
                    start_time = time.time()
                    while (
                        time.time() - start_time
                    ) < interval and dynamic_state.should_stream and dynamic_state.in_cot:
                        time.sleep(0.005)

        except Exception as e:
            if str(e) == "list index out of range":
                dynamic_state.stream_completed = True                
            else:
                if str(e) == "The read operation timed out":
                    error_msg = lang_data["api_interrupted"]
                else:
                    error_msg = "❓ " + str(e)
                # full_response += f"\n\n[{error_msg}: {str(e)}]"
                editor_label_error = f"{lang_data['editor_label']} - {error_msg}"
                self.is_error = True
                dynamic_state.label_passthrough = True


        finally:
            dynamic_state.should_stream = False
            if "response_stream" in locals():
                response_stream.close()
            final_status = (
                lang_data["completed"]
                if dynamic_state.stream_completed
                else lang_data["interrupted"]
            )
            editor_label = f"{lang_data['editor_label']} - {final_status}"
            if not self.is_error:
                yield self.current["cot"] + ("</think>" if not dynamic_state.in_cot else ""), gr.update(label=editor_label), self.flatten_output()
            else:
                yield self.current["cot"] + ("</think>" if not dynamic_state.in_cot else ""), gr.update(label=editor_label_error), self.flatten_output() + [
                {
                    "role": "assistant",
                    "content": error_msg,
                    "metadata": {"title": f"❌Error"},
                }
            ]
            self.is_error =  False
            


def update_interface_language(selected_lang, convo_state, dynamic_state):
    """更新界面语言配置"""
    convo_state.current_language = selected_lang
    dynamic_state.current_language = selected_lang
    lang_data = LANGUAGE_CONFIG[selected_lang]
    base_editor_label = lang_data["editor_label"]
    status_suffix = (
        lang_data["completed"]
        if dynamic_state.stream_completed
        else lang_data["interrupted"]
    )
    editor_label = f"{base_editor_label} - {status_suffix}"
    return [
        gr.update(value=f"{lang_data['title']}"),
        gr.update(
            label=lang_data["prompt_label"], placeholder=lang_data["prompt_placeholder"]
        ),
        gr.update(label=editor_label, placeholder=lang_data["editor_placeholder"]),
        gr.update(
            label=lang_data["sync_threshold_label"],
            info=lang_data["sync_threshold_info"],
        ),
        gr.update(
            label=lang_data["throughput_label"], info=lang_data["throughput_info"]
        ),
        gr.update(
            value=(
                lang_data["pause_btn"]
                if dynamic_state.should_stream
                else lang_data["generate_btn"]
            ),
            variant="secondary" if dynamic_state.should_stream else "primary",
        ),
        gr.update(label=lang_data["language_label"]),
        gr.update(
            value=lang_data["clear_btn"], interactive=not dynamic_state.should_stream
        ),
        gr.update(value=lang_data["introduction"]),
        gr.update(value=lang_data["bot_default"], label=lang_data["bot_label"]),
    ]


theme = gr.themes.Base(font="system-ui", primary_hue="stone")

with gr.Blocks(theme=theme, css_paths="styles.css") as demo:
    convo_state = gr.State(ConvoState)
    dynamic_state = gr.State(DynamicState)

    bot_default = LANGUAGE_CONFIG["en"]["bot_default"] + [
                    {
                        "role": "assistant",
                        "content": f"Running `{os.getenv('API_MODEL')}` @ {os.getenv('API_URL')}  \n Performance subjects to API provider situation",
                        "metadata": {"title": f"API INFO"},
                    }
                ]

    with gr.Row(variant=""):
        title_md = gr.Markdown(
            f"{LANGUAGE_CONFIG['en']['title']} ",
            container=False,
        )
        lang_selector = gr.Dropdown(
            choices=["en", "zh"],
            value="en",
            elem_id="compact_lang_selector",
            scale=0,
            container=False,
        )

    with gr.Row(equal_height=True):

        with gr.Column(scale=1, min_width=400):
            prompt_input = gr.Textbox(
                label=LANGUAGE_CONFIG["en"]["prompt_label"],
                lines=2,
                placeholder=LANGUAGE_CONFIG["en"]["prompt_placeholder"],
                max_lines=5,
            )
            thought_editor = gr.Textbox(
                label=f"{LANGUAGE_CONFIG['en']['editor_label']} - {LANGUAGE_CONFIG['en']['editor_default']}",
                lines=16,
                placeholder=LANGUAGE_CONFIG["en"]["editor_placeholder"],
                autofocus=True,
                elem_id="editor",
            )
            with gr.Row():
                control_button = gr.Button(
                    value=LANGUAGE_CONFIG["en"]["generate_btn"], variant="primary"
                )
                next_turn_btn = gr.Button(
                    value=LANGUAGE_CONFIG["en"]["clear_btn"], interactive=True
                )

        with gr.Column(scale=1, min_width=500):
            chatbot = gr.Chatbot(
                type="messages",
                height=300,
                value=bot_default,
                group_consecutive_messages=False,
                show_copy_all_button=True,
                show_share_button=True,
                label=LANGUAGE_CONFIG["en"]["bot_label"],
            )
            with gr.Row():
                sync_threshold_slider = gr.Slider(
                    minimum=0,
                    maximum=20,
                    value=AppConfig.SYNC_THRESHOLD_DEFAULT,
                    step=1,
                    label=LANGUAGE_CONFIG["en"]["sync_threshold_label"],
                    info=LANGUAGE_CONFIG["en"]["sync_threshold_info"],
                )
                throughput_control = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=AppConfig.DEFAULT_THROUGHPUT,
                    step=1,
                    label=LANGUAGE_CONFIG["en"]["throughput_label"],
                    info=LANGUAGE_CONFIG["en"]["throughput_info"],
                )

            intro_md = gr.Markdown(LANGUAGE_CONFIG["en"]["introduction"], visible=False)

    # 交互逻辑
    stateful_ui = (control_button, thought_editor, next_turn_btn)

    throughput_control.change(
        lambda val, s: setattr(s, "throughput", val),
        [throughput_control, convo_state],
        None,
        concurrency_limit=None,
    )

    sync_threshold_slider.change(
        lambda val, s: setattr(s, "sync_threshold", val),
        [sync_threshold_slider, convo_state],
        None,
        concurrency_limit=None,
    )

    def wrap_stream_generator(convo_state, dynamic_state, prompt, content):
        for response in convo_state.generate_ai_response(
            prompt, content, dynamic_state
        ):
            yield response

    gr.on(
        [control_button.click, prompt_input.submit, thought_editor.submit],
        lambda d: d.control_button_handler(),
        [dynamic_state],
        stateful_ui,
        show_progress=False,
        concurrency_limit=None,
    ).then(
        wrap_stream_generator,
        [convo_state, dynamic_state, prompt_input, thought_editor],
        [thought_editor, thought_editor, chatbot],
        concurrency_limit=1000,
    ).then(
        lambda d: d.ui_state_controller(),
        [dynamic_state],
        stateful_ui,
        show_progress=False,
        concurrency_limit=None,
    )

    next_turn_btn.click(
        lambda d: d.reset_workspace(),
        [dynamic_state],
        stateful_ui + (thought_editor, prompt_input, chatbot),
        concurrency_limit=None,
        show_progress=False
    )

    lang_selector.change(
        lambda lang, s, d: update_interface_language(lang, s, d),
        [lang_selector, convo_state, dynamic_state],
        [
            title_md,
            prompt_input,
            thought_editor,
            sync_threshold_slider,
            throughput_control,
            control_button,
            lang_selector,
            next_turn_btn,
            intro_md,
            chatbot,
        ],
        concurrency_limit=None,
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1000)
    demo.launch()
