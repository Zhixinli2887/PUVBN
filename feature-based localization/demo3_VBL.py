import os
import cv2
import time
import numpy as np
import tkinter as tk
from PIL import Image
from PIL import ImageTk
from tkinter import ttk
import tkinter.filedialog
import tkinter.scrolledtext
from py360convert import e2p
from transformers import CLIPModel, AutoProcessor
from sklearn.metrics.pairwise import cosine_similarity


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Vision-based Localization V1.0')
        self.workspace_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        self.GSV_feat_dir = os.path.join(self.workspace_dir, 'GSV_feat')
        self.GSV_dir = os.path.join(self.workspace_dir, 'GSV')
        self.query_fp = os.path.join(self.workspace_dir, 'temp\query.png')
        self.query_source, self.query_feat = None, None
        self.GSV_fnames, self.GSV_fps, self.GSV_feat_fps, self.GSV_feats = None, None, None, None
        self.scores, self.uv_ids = [], []
        self.out_img_size, self.FOV = (300, 480), (74, 50)
        self.query_imageTK, self.GSV_imgTK, self.GSV_patchTK = None, None, None

        # App Frames
        self.setting_frame = tk.LabelFrame(self, text="Setting View")
        self.table_frame = tk.LabelFrame(self, text="Table View")
        self.display_frame = tk.LabelFrame(self, text="Visualization View")
        self.GSV_frame = tk.LabelFrame(self, text="Localization View")
        self.GSV_patch_frame = tk.LabelFrame(self, text="Projected GSV View")

        self.setting_frame.grid(row=0, column=0, padx=5, pady=5, sticky='n')
        self.table_frame.grid(row=0, column=1, padx=5, pady=5, sticky='n')
        self.display_frame.grid(row=0, column=2, padx=5, pady=5, sticky='n')
        self.GSV_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky='n')
        self.GSV_patch_frame.grid(row=1, column=2, columnspan=1, padx=5, pady=5, sticky='n')

        # Button Frame
        self.open_dir_btn = tk.Button(self.setting_frame, text='Open Workspace', command=self.open_folder)
        self.open_dir_btn.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        self.open_dir_text = tk.Entry(self.setting_frame)
        self.open_dir_text.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        self.open_query_btn = tk.Button(self.setting_frame, text='Open Query Image', command=self.open_query)
        self.open_query_btn.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

        self.open_query_text = tk.Entry(self.setting_frame)
        self.open_query_text.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

        self.run_btn = tk.Button(self.setting_frame, text='Run', command=self.process)
        self.run_btn.grid(row=2, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

        self.process_bar = ttk.Progressbar(self.setting_frame, mode='determinate')
        self.process_bar.grid(row=3, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

        self.text_area = tk.scrolledtext.ScrolledText(self.setting_frame, wrap=tk.WORD, width=40, height=10,
                                                      state='disabled')
        self.text_area.grid(row=4, column=0, columnspan=2, sticky="ewn", padx=5, pady=5)
        self.write_log('*******Vision-based Localization*******\n')
        self.write_log(f'Date: {time.ctime()}\n')
        self.write_log('Please select the workspace for localization...\n')

        self.open_dir_text.delete(0, tk.END)
        self.open_dir_text.insert(0, self.workspace_dir)

        self.open_query_text.delete(0, tk.END)
        self.open_query_text.insert(0, self.query_fp)

        # Table Frame
        self.table_view = ttk.Treeview(self.table_frame, columns=('U', 'V', 'S'), height=14, selectmode="browse")
        self.table_view.grid(row=0, column=0, sticky="ewn", padx=5, pady=5)
        self.table_view.heading('#0', text='GSV_ID')
        self.table_view.column("# 0", anchor='center', stretch=False, width=80)
        self.table_view.heading('U', text='U')
        self.table_view.column('U', anchor='center', stretch=False, width=80)
        self.table_view.heading('V', text='V')
        self.table_view.column('V', anchor='center', stretch=False, width=80)
        self.table_view.heading('S', text='Similarity')
        self.table_view.column('S', anchor='center', stretch=False, width=80)
        self.table_view.bind("<Double-1>", self.draw_GSV)

        self.table_scroll = ttk.Scrollbar(self.table_frame, orient="vertical", command=self.table_view.yview)
        self.table_scroll.grid(row=0, column=1, sticky="ns")
        self.table_view.configure(yscrollcommand=self.table_scroll.set)

        # Display frame
        self.image_display = tk.Label(self.display_frame, text='Image Display Area')
        self.image_display.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.draw_query()

        # GSV frame
        self.GSV_display = tk.Label(self.GSV_frame, text='GSV Display Area')
        self.GSV_display.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # GSV patch frame
        self.GSV_patch_display = tk.Label(self.GSV_patch_frame, text='Projected GSV Display Area')
        self.GSV_patch_display.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Load models
        self.write_log('Loading CLIP model...\n')
        self.model_CLIP = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.load_GSV_feat()
        self.draw_GSV()

        App.resizable(self, width=0, height=0)


    def open_folder(self):
        folder_path = tkinter.filedialog.askdirectory(title='Select a folder', initialdir=self.workspace_dir)
        if folder_path:
            self.open_dir_text.delete(0, tk.END)
            self.open_dir_text.insert(0, folder_path)
            self.write_log(f'Folder selected: {folder_path}\n')
            self.load_GSV_feat()
        else:
            self.write_log('No folder selected.\n')


    def get_img_feat(self):
        inputs = self.processor(images=self.query_source, return_tensors="pt")
        feat = self.model_CLIP.get_image_features(**inputs)
        return feat.cpu().detach().numpy()[0]


    def open_query(self):
        file_path = tkinter.filedialog.askopenfilename(title='Select a file', initialdir=self.workspace_dir)
        if file_path:
            self.open_query_text.delete(0, tk.END)
            self.open_query_text.insert(0, file_path)
            self.write_log(f'File selected: {file_path}\n')
            self.query_fp = file_path
        else:
            self.write_log('No file selected.\n')
        self.draw_query()
        self.load_GSV_feat()


    def write_log(self, s):
        self.text_area.configure(state='normal')
        self.text_area.insert('end', s)
        self.text_area.configure(state='disabled')
        self.text_area.see('end')


    def draw_GSV(self, event=None):
        try:
            idx = self.table_view.index(self.table_view.selection()[0])
        except:
            idx = 0

        GSV_fp = os.path.join(self.GSV_dir, self.GSV_fnames[idx])
        GSV_img = cv2.cvtColor(cv2.imread(GSV_fp), cv2.COLOR_BGR2RGB)

        if len(self.uv_ids) == 0:
            u, v = 0, 0
        else:
            u, v = self.uvs[self.uv_ids[idx]]
        GSV_patch = Image.fromarray(e2p(GSV_img, self.FOV, u, v, self.out_img_size))

        # Projection center from uv of GSV image
        uc, vc = int((u + 180) / 360 * GSV_img.shape[1]), int((90 - v) / 180 * GSV_img.shape[0])
        GSV_img = cv2.circle(GSV_img, (uc, vc), 100, (0, 0, 0), -1)
        GSV_img = cv2.rectangle(GSV_img, (uc - 960, vc - 540), (uc + 960, vc + 540), (0, 0, 0), 50)

        GSV_img = Image.fromarray(GSV_img).resize((600, 300))
        self.GSV_imgTK = ImageTk.PhotoImage(GSV_img)
        self.GSV_patchTK = ImageTk.PhotoImage(GSV_patch)
        self.GSV_display.configure(image=self.GSV_imgTK)
        self.GSV_patch_display.configure(image=self.GSV_patchTK)


    def draw_query(self):
        self.query_source = cv2.imread(self.query_fp)
        image = Image.fromarray(self.query_source)
        image = image.resize((480, 300))
        self.query_imageTK = ImageTk.PhotoImage(image)
        self.image_display.configure(image=self.query_imageTK)


    def load_GSV_feat(self):
        for row in self.table_view.get_children():
            self.table_view.delete(row)

        self.write_log('Loading GSV features...\n')
        self.GSV_fnames = [item for item in os.listdir(self.GSV_dir)]
        self.GSV_fps = [os.path.join(self.GSV_dir, item) for item in os.listdir(self.GSV_dir)]
        self.GSV_feat_fps = [os.path.join(self.GSV_feat_dir, item.split('.')[0] + '.npy') for item in self.GSV_fnames]
        self.GSV_feats, self.uvs, idx = [], [], None

        for i in range(len(self.GSV_fps)):
            self.process_bar['value'] = int(100 * (i + 1) / len(self.GSV_fps))
            self.table_view.insert('', 'end', text=f'{self.GSV_fnames[i]}',
                                   values=[f'-', f'-', f'-'])
            feat = np.load(self.GSV_feat_fps[i], allow_pickle=True)
            if len(self.uvs) == 0:
                idx = np.where(feat[:, 1] <= 15)[0]
                self.uvs = feat[idx, 0:2]
            feat = feat[idx, 2:]
            self.GSV_feats.append(feat)
        self.GSV_feats = np.array(self.GSV_feats)
        self.process_bar['value'] = 0


    def process(self):
        time_start = time.time()
        for row in self.table_view.get_children():
            self.table_view.delete(row)

        self.write_log('Processing...\n')
        self.write_log('Converting query image to feature...\n')
        self.process_bar['value'] = 0
        self.query_feat = self.get_img_feat()
        self.process_bar['value'] = 10
        self.write_log('Computing similarity...\n')
        self.scores, self.uv_ids = [], []

        for i in range(len(self.GSV_feats)):
            scores = cosine_similarity(self.query_feat.reshape(1, -1), self.GSV_feats[i])
            score = np.max(scores)
            uv_index = np.argmax(scores)
            self.scores.append(score)
            self.uv_ids.append(uv_index)
            self.process_bar['value'] = int(10 + 90 * (i + 1) / len(self.GSV_feats))

        ids_sorted = np.argsort(self.scores)[::-1]
        self.GSV_fnames = [self.GSV_fnames[i] for i in ids_sorted]
        self.scores = np.array(self.scores)[ids_sorted]
        self.uv_ids = np.array(self.uv_ids)[ids_sorted]

        for i in range(len(ids_sorted)):
            score = self.scores[i]
            u, v = self.uvs[self.uv_ids[i]]

            # Update the table
            self.table_view.insert('', 'end', text=f'{self.GSV_fnames[i]}',
                                   values=[f'{u}°', f'{v}°', f'{score:.4f}'])

        time_end = time.time()
        self.write_log(f'Processing completed in {time_end - time_start:.2f} seconds.\n')
        self.draw_GSV()


if __name__ == "__main__":
    win = App()
    win.mainloop()