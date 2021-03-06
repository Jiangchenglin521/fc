Open-Domain Free Chatbot
=========================================

功能描述
---

本项目给出一个闲聊型聊天模型，其属于被动交互型，即由用户发起对话，机器理解对话并作出相应的回应。主要进行客观性讨论和无明确主题的互动，情感感知。主要用于辅助其他事实任务型问答模型进行更可靠地人机交互，提升用户体验。

开发环境
---

* 主要依赖的工具包以及版本，详见requirements.txt。

项目结构
---

* fc/config: 项目文件读取以及文件存储路径信息配置模块。
* fc/chatbot.py: 模型训练、测试模块，模型文件存储等，并且文件开头设置全局参数设置接口函数，方便使用。
* fc/data_utils.py: 数据预处理模块，包括数据清洗分词，词典创建，词向量创建等模块。
* fc/seq2seq_model.py: 主模型创建模块，数据封装模块。
* fc/seq2seq.py and rnn_cell.py: 主模型中各子模块封装函数所在文件，如encoder，decoder,loss等。

数据格式和文件简要说明
---

* 训练和测试数据格式一致，raw_data形式为[[post, label],[response, label]].形式如： **[["你 的 也 不 算 长 啊\n", 0], ["现在 已经 长 了 好 不好 。 军训 要 剪 短发 啦 。\n", 1]]**, 两个label可认为是占位符，当前闲聊模型不起作用，可用 0 代替。
* 模型文件保存在fc/train下面。数据文件在fc/data下面
* 注：函数掺杂emotion标识处理部分，请忽略，已经略过！！
* 已更新成最新的ITF-SCE目标训练版本。

使用方法
---

* 在配置文件里配置号文件路径，将*训练和验证*数据放入data文件里。设置好训练的epoch。
* 训练：运行 python3 chatbot.py ，同时可以记录log文件，有训练和验证同步的ppx信息。
* 预测：运行python3 chatbot.py --decode  进行预测使用。运行 --evaluation 进行评估。
