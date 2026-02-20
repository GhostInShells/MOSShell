import os
import subprocess
import sys
from pathlib import Path

import fitz  # PyMuPDF
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QTextEdit,
    QFileDialog,
    QMessageBox,
)

from ghoshell_common.contracts import FileStorage
from ghoshell_common.helpers import timestamp_ms


class ConvertThread(QThread):
    """转换工作线程"""

    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, pptx_path, output_dir):
        super().__init__()
        self.pptx_path = pptx_path
        self.output_dir = output_dir

    def run(self):
        try:
            self.log_signal.emit("🔄 开始转换流程...")
            self.log_signal.emit(f"源文件: {self.pptx_path}")
            self.log_signal.emit(f"输出目录: {self.output_dir}")

            image_paths = convert_pptx_to_pngs(
                self.pptx_path, output_img_dir=self.output_dir, log_callback=self.log_signal.emit
            )

            self.log_signal.emit(f"✅ 转换成功！共生成 {len(image_paths)} 张图片")
            self.finished_signal.emit(True, f"成功生成 {len(image_paths)} 张图片")
        except Exception as e:
            self.log_signal.emit(f"❌ 转换失败: {str(e)}")
            self.finished_signal.emit(False, str(e))


DEFAULT_MD = """
---
title: ""
outline: ""
---
# 演讲词

# FAQ
""".lstrip()

DEFAULT_META = """
name: "{name}"
description: "{description}"
origin_filetype: "{origin_filetype}"
origin_filepath: "{origin_filepath}"
created_at: {created_at}
updated_at: {updated_at}
""".strip()


def convert_pptx_to_pngs(pptx_path, output_img_dir, log_callback=print):
    """
    Mac系统下将PPTX每页转为PNG图片（PDF中转方案）
    :param pptx_path: PPTX文件路径
    :param output_img_dir: 图片输出目录（自动创建）
    :param log_callback: 日志回调函数
    :return: 生成的图片路径列表
    """
    if not os.path.exists(pptx_path):
        raise FileNotFoundError(f"PPTX文件不存在：{pptx_path}")

    os.makedirs(output_img_dir, exist_ok=True)

    # ---------- 1. PPTX → PDF ----------
    log_callback("步骤1/2：使用LibreOffice转换为PDF...")
    libreoffice_path = "/Applications/LibreOffice.app/Contents/MacOS/soffice"
    if not os.path.exists(libreoffice_path):
        raise RuntimeError(
            f"未找到LibreOffice，请确认路径：{libreoffice_path}，或者执行 brew install --cask libreoffice 安装依赖"
        )

    pdf_filename = Path(pptx_path).stem + ".pdf"
    pdf_path = os.path.join(output_img_dir, pdf_filename)

    cmd = [libreoffice_path, "--headless", "--convert-to", "pdf", "--outdir", output_img_dir, pptx_path]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"LibreOffice转换PDF失败：{e.stderr}")
    except Exception as e:
        raise RuntimeError(f"LibreOffice调用异常：{str(e)}")

    # ---------- 2. PDF → PNG ----------
    log_callback("步骤2/2：使用PyMuPDF将PDF每一页转为PNG...")
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise RuntimeError(f"无法打开PDF文件，请检查PyMuPDF安装：{str(e)}")

    meta_yaml = os.path.join(output_img_dir, ".meta.yaml")
    with open(meta_yaml, "w") as _meta:
        _meta.write(
            DEFAULT_META.format(
                name=Path(pptx_path).stem,
                description="",
                origin_filetype=Path(pptx_path).suffix,
                origin_filepath=pptx_path,
                created_at=timestamp_ms(),
                updated_at=timestamp_ms(),
            )
        )

    image_paths = []
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        output_file = os.path.join(output_img_dir, f"slide_{page_num + 1:03d}.png")
        pix.save(output_file)

        description_md = output_file + ".md"
        with open(description_md, "w") as _md:
            _md.write(DEFAULT_MD)

        image_paths.append(output_file)
        log_callback(f"   已生成: {os.path.basename(output_file)}")

    doc.close()
    os.remove(pdf_path)  # 如需删除临时PDF，取消注释

    log_callback("转换完成！")
    return image_paths


class PPTXConverterWindow(QMainWindow):
    def __init__(self, studio_storage: FileStorage):
        # ---------- 环境初始化 ----------
        self.studio_storage = studio_storage

        super().__init__()
        self.setWindowTitle("PPTX 转 PNG 工具 (输入文件夹名)")
        self.setMinimumSize(750, 500)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # ---------- 1. PPTX文件选择 ----------
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("PPTX文件:"))
        self.pptx_path_edit = QLineEdit()
        self.pptx_path_edit.setPlaceholderText("请选择PPTX文件...")
        file_layout.addWidget(self.pptx_path_edit)
        btn_browse = QPushButton("浏览...")
        btn_browse.clicked.connect(self.on_browse_pptx)
        file_layout.addWidget(btn_browse)
        main_layout.addLayout(file_layout)

        # ---------- 2. 输出文件夹名输入（无浏览按钮）----------
        folder_layout = QHBoxLayout()
        folder_layout.addWidget(QLabel("输出文件夹名:"))
        self.folder_name_edit = QLineEdit()
        self.folder_name_edit.setText("example")  # 默认名称
        self.folder_name_edit.setPlaceholderText("例如：example")
        folder_layout.addWidget(self.folder_name_edit)

        # 显示完整输出路径（只读，自动更新）
        self.full_path_label = QLabel()
        self.full_path_label.setStyleSheet("background-color: #f5f5f5; padding: 4px; border: 1px solid #ddd;")
        folder_layout.addWidget(QLabel("完整路径:"))
        folder_layout.addWidget(self.full_path_label, 1)
        main_layout.addLayout(folder_layout)

        # 连接输入变化事件，实时更新完整路径
        self.folder_name_edit.textChanged.connect(self.update_full_path)

        # ---------- 显示基础路径（只读）----------
        base_path_layout = QHBoxLayout()
        base_path_layout.addWidget(QLabel("基础输出目录:"))
        base_path_label = QLabel(self.studio_storage.abspath())
        base_path_label.setStyleSheet("background-color: #f0f0f0; padding: 4px; border: 1px solid #ccc;")
        base_path_layout.addWidget(base_path_label, 1)
        main_layout.addLayout(base_path_layout)

        # ---------- 转换按钮 ----------
        self.btn_convert = QPushButton("开始转换")
        self.btn_convert.clicked.connect(self.on_convert)
        self.btn_convert.setStyleSheet("QPushButton { font-weight: bold; padding: 8px; }")
        main_layout.addWidget(self.btn_convert)

        # ---------- 日志显示 ----------
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFontFamily("Monospace")
        main_layout.addWidget(self.log_text)

        self.thread = None
        self.update_full_path()  # 初始化完整路径显示

    def update_full_path(self):
        """根据用户输入的文件夹名更新完整输出路径预览"""
        folder_name = self.folder_name_edit.text().strip()
        if not folder_name:
            folder_name = "example"  # 默认
        full_path = os.path.join(self.studio_storage.abspath(), folder_name)
        self.full_path_label.setText(full_path)

    def on_browse_pptx(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择PPTX文件", "", "PPTX文件 (*.pptx);;所有文件 (*.*)")
        if file_path:
            self.pptx_path_edit.setText(file_path)

    def on_convert(self):
        pptx_path = self.pptx_path_edit.text().strip()
        if not pptx_path:
            QMessageBox.warning(self, "警告", "请先选择PPTX文件！")
            return
        if not os.path.exists(pptx_path):
            QMessageBox.critical(self, "错误", "PPTX文件不存在，请重新选择。")
            return

        # 获取用户输入的文件夹名，构建完整输出路径
        folder_name = self.folder_name_edit.text().strip()
        if not folder_name:
            folder_name = "example"
        output_dir = os.path.join(self.studio_storage.abspath(), folder_name)

        # 禁用按钮
        self.btn_convert.setEnabled(False)
        self.log_text.clear()

        # 启动工作线程
        self.thread = ConvertThread(pptx_path, output_dir)
        self.thread.log_signal.connect(self.append_log)
        self.thread.finished_signal.connect(self.on_convert_finished)
        self.thread.start()

    def append_log(self, message):
        self.log_text.append(message)

    def on_convert_finished(self, success, msg):
        self.btn_convert.setEnabled(True)
        if success:
            QMessageBox.information(self, "完成", f"转换成功！\n{msg}")
        else:
            QMessageBox.critical(self, "转换失败", f"错误信息：\n{msg}")

    def closeEvent(self, event):
        if self.thread and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait(1000)
        event.accept()
