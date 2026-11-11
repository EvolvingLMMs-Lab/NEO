from huggingface_hub import snapshot_download
from ..smp import *
from .video_base import VideoBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
import json


FAIL_MSG = "Failed to obtain answer via API."
tsv_data_path = ""


def seconds_to_minutes_secondswithdot(seconds):
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{int(minutes):02d}:{secs:05.2f}"


def seconds_to_minutes_seconds(seconds):
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{int(minutes):02d}:{int(secs):02d}"


class LVBench(VideoBaseDataset):

    MD5 = "bfc25490be4080aa5494b883370b6b1f"

    BASE_SYS = "Carefully watch this video and pay attention to every detail. "
    SYS = (
        BASE_SYS
        + "Based on your observations, select the best option that accurately addresses the question."
    )

    FRAMES_TMPL_NOSUB = """
These are the frames of a video. \
Select the best answer to the following multiple-choice question based on the video. \
Respond with only the letter (A, B, C, or D) of the correct option.
"""

    FRAMES_TMPL_SUB = """
These are the frames of a video. \
This video's subtitles are listed below:
{}
Select the best answer to the following multiple-choice question based on the video. \
Respond with only the letter (A, B, C, or D) of the correct option.
"""

    FRAMES_TMPL_AUDIO = """
These are the frames of a video and the corresponding audio. \
Select the best answer to the following multiple-choice question based on the video. \
Respond with only the letter (A, B, C, or D) of the correct option.
"""

    TYPE = "Video-MCQ"

    def __init__(self, dataset="LVBench", use_audio=False, nframe=-1, fps=-1):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)
        self.use_audio = use_audio
        self.dataset_name = dataset
        self.dataset_path = ""

        # assert not (self.use_subtitle and self.use_audio), 'Cannot use both subtitle and audio at the same time.'

    @classmethod
    def supported_datasets(cls):
        return ["LVBench"]

    def prepare_dataset(self, repo_id="AIWinter/LVBench"):
        data_file = osp.join(dataset_path, f"{dataset_name}.tsv")
        return dict(data_file=data_file, root="")

    def save_video_frames(self, video, video_llm=False):

        vid_path = video
        suffix = video.split(".")[-1]
        video = video.replace(f".{suffix}", "")
        video = (
            ""
            + video.split("/")[-1]
        )
        # osp.join(self.data_root, 'videos', video + '.mp4')
        import decord

        vid = decord.VideoReader(vid_path)
        video_info = {
            "fps": vid.get_avg_fps(),
            "n_frames": len(vid),
        }
        if self.nframe > 0 and self.fps < 0:
            step_size = len(vid) / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(video)
        elif self.fps > 0:
            # not constrained by num_frames, get frames by fps
            total_duration = video_info["n_frames"] / video_info["fps"]
            required_frames = int(total_duration * self.fps)
            ## constrained by num_frames
            max_frames = 256
            if required_frames <= max_frames:
                required_frames = int(total_duration * self.fps)
                step_size = video_info["fps"] / self.fps
            else:
                required_frames = max_frames
                step_size = len(vid) / (max_frames + 1)

            # total_duration = video_info['n_frames'] / video_info['fps']
            # required_frames = int(total_duration * self.fps)
            # step_size = video_info['fps'] / self.fps

            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(video, len(indices))

        flag = np.all([osp.exists(p) for p in frame_paths])

        if not flag:
            images = [vid[i].asnumpy() for i in indices]
            images = [Image.fromarray(arr) for arr in images]
            for im, pth in zip(images, frame_paths):
                if not osp.exists(pth) and not video_llm:
                    im.save(pth)

        return frame_paths, indices, video_info

    def build_prompt(self, line, video_llm):
        # line是tsv的line 看看prompt咋拼的
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        # message = []
        # [dict(type='text', value=self.SYS)]

        if video_llm:
            message = []
            message.append(dict(type="video", value=line["video"]))

        else:
            frames, indices, video_info = self.save_video_frames(
                line["video"], video_llm
            )

            video_duration = video_info["n_frames"] / video_info["fps"]
            sample_fps = len(indices) / video_duration
            temporal_instruction = f"The time range of this video is [00:00-{seconds_to_minutes_seconds(video_duration)}], and the following is a series of {len(frames)} frames sampled at {round(sample_fps,1)} FPS.\n"
            sample_timestamp = [idx / video_info["fps"] for idx in indices]
            message = [dict(type="text", value=temporal_instruction, role="temporal")]
            for idx, im in enumerate(frames):
                message.append(dict(type="image", value=im))
                message.append(
                    dict(
                        type="text",
                        value=seconds_to_minutes_secondswithdot(sample_timestamp[idx]),
                        role="timestamp",
                    )
                )

            # for im in frames:
            #     message.append(dict(type='image', value=im))
            # if self.use_audio:
            #     message.append(dict(type='audio', value=osp.join(self.data_root, 'audios', line['video'] + '.wav')))
        text_prompt = line["question"] + "\nAnswer the question with the option letter."
        message.append(dict(type="text", value=text_prompt))
        # question_str = line['question'] + '\n' + '\n'.join(eval(line['candidates']))
        # prompt = 'Question: {}\nAnswer: '.format(question_str)
        # message.append(dict(type='text', value=prompt))
        return message

    # It returns a dictionary
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.worldsense import (
            get_dimension_rating,
            extract_characters_regex,
            extract_option,
        )

        assert eval_file.endswith(".xlsx"), "data file should be an xlsx file"

        tmp_file = eval_file.replace(".xlsx", "_tmp.pkl")
        tgt_file = eval_file.replace(".xlsx", "_rating.json")
        score_file = eval_file.replace(".xlsx", "_score.xlsx")

        if not osp.exists(score_file):
            model = "exact_matching"
            # judge_kwargs.get('model', 'exact_matching')
            assert model in ["chatgpt-0125", "exact_matching", "gpt-4-0125"]

            if model == "exact_matching":
                model = None
            elif gpt_key_set():
                model = build_judge(**judge_kwargs)
                if not model.working():
                    warnings.warn(
                        "OPENAI API is not working properly, will use exact matching for evaluation"
                    )
                    warnings.warn(DEBUG_MESSAGE)
                    model = None
            else:
                warnings.warn(
                    "OPENAI_API_KEY is not set properly, will use exact matching for evaluation"
                )
                model = None
            res = {} if not osp.exists(tmp_file) else load(tmp_file)
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

            data = load(eval_file)
            data_un = data[~pd.isna(data["prediction"])]

            for idx in data["index"]:
                ans = data.loc[data["index"] == idx, "answer"].values[0]
                pred = str(data.loc[data["index"] == idx, "prediction"].values[0])

                if extract_characters_regex(pred) == "":
                    extract_pred = extract_option(
                        model,
                        data.loc[data["index"] == idx].to_dict(orient="records")[0],
                        "WorldSense",
                    )
                    data.loc[idx, "score"] = int(extract_pred == ans)
                else:
                    data.loc[idx, "score"] = int(extract_characters_regex(pred) == ans)

            rejected = [x for x in data["score"] if x == -1]
            correct = [x for x in data["score"] if x == 1]
            print("Correct Rate:", len(correct) / len(data))
            print(
                f"Among {len(data)} questions, failed to obtain prediction for {len(data) - len(data_un)} questions, "
                f"failed to obtain the score for another {len(rejected)} questions. "
                f"Those questions will be counted as -1 score in ALL rating, and will not be counted in VALID rating."
            )

            dump(data, score_file)

            # def get_dimension_rating(score_file):
            data = load(score_file)
            """
            q_type_stat_dict = defaultdict(defaultdict(int))
            # 按照 task type统计
            for i in range(len(data)):
                question_type_list = eval(data.iloc[i]['question_type'])
                is_correct = data.iloc[i]['score']
                #is_correct = data.iloc[i]['']
                for q_type in question_type_list:
                    q_type_stat_dict[q_type]['total'] +=1
                    if int(is_correct) == 1:
                        q_type_stat_dict[q_type]['correct'] +=1
            """

            # return q_type_stat_dict

        # rating = get_dimension_rating(score_file)
        # dump(rating, tgt_file)
        # return rating
