from tkinter import *
from tkinter.ttk import *
import Logic
from functools import partial
window = Tk()

window.title("Speaker Recognition")
window.geometry('460x300')

## status label
status_line = 75
lbl = Label(window, text="Click 'Start Recording' to record audio")
lbl.place(x = 130, y = status_line)


## Predicted speaker label
status_line = 175
pred_spk_lbl = Label(window, text="Click 'Start Recording' to record audio")
pred_spk_lbl.place(x = 130, y = status_line)

## Recording Buttons
rec_line = 20
# Start Recording button
strt_rec = Button(window, text="Start Recording",width=15, command=partial(Logic.start_recording,window,lbl))
strt_rec.place(x=30, y=rec_line)

# Play Recording button
play = Button (window, text="Play Recording",width=15, command=Logic.play_audio)
play.place(x=330, y=rec_line)

##predict speaker
predict_btn_line = 120
predict_btn = Button (window, text="Predict Speaker",width=15, command=Logic.start_recording)
predict_btn.place(x=180, y=predict_btn_line)

window.mainloop()