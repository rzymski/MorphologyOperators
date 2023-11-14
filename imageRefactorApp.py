from tkinter import *
# import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
from tkinter.filedialog import asksaveasfilename, askopenfilename
import time
import tkinter.font as font
from copy import deepcopy
import re
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic


class ImageRefactorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image refactor Piotr Szumowski")
        bigFont = font.Font(size=12, weight="bold")
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        self.root.geometry(f"{self.screen_width}x{self.screen_height}")
        self.frame = LabelFrame(self.root, padx=0, pady=0, labelanchor="w")
        self.frame.pack(side="left", fill="both")
        # Button to load jpg image
        self.loadJPGButton = Button(self.frame, text="Load JPG", command=self.loadJPG, padx=10, pady=10)
        self.loadJPGButton.grid(row=0, column=0, sticky="WE")
        self.loadJPGButton['font'] = bigFont
        # Button to reload previously loaded jpg image
        self.reloadOriginalJPGButton = Button(self.frame, text="Reload original JPG", command=self.reloadOriginalJPG, padx=10, pady=10)
        self.reloadOriginalJPGButton.grid(row=1, column=0, sticky="WE")
        self.reloadOriginalJPGButton['font'] = bigFont
        # Button to save image
        self.saveJPGButton = Button(self.frame, text="Save JPG", command=self.saveJPG, padx=10, pady=10)
        self.saveJPGButton.grid(row=2, column=0, sticky="WE")
        self.saveJPGButton['font'] = bigFont
        # LabelFrame for pixel
        self.pixelInfoLabel = LabelFrame(self.frame, text="Pixel", padx=0, pady=0, labelanchor="nw")
        self.pixelInfoLabel.grid(row=3, column=0, sticky="WE")
        # Labels for pixel
        self.pixelXLabel = Label(self.pixelInfoLabel, text="X")
        self.pixelXLabel.grid(row=0, column=0, sticky="E")
        self.pixelYLabel = Label(self.pixelInfoLabel, text="Y")
        self.pixelYLabel.grid(row=1, column=0, sticky="E")
        self.pixelRedLabel = Label(self.pixelInfoLabel, text="Red")
        self.pixelRedLabel.grid(row=2, column=0, sticky="E")
        self.pixelGreenLabel = Label(self.pixelInfoLabel, text="Green")
        self.pixelGreenLabel.grid(row=3, column=0, sticky="E")
        self.pixelBlueLabel = Label(self.pixelInfoLabel, text="Blue")
        self.pixelBlueLabel.grid(row=4, column=0, sticky="E")
        # Entries for pixel
        self.pixelXEntry = Entry(self.pixelInfoLabel, state=DISABLED, disabledforeground="black", disabledbackground="white", justify=CENTER)
        self.pixelXEntry.grid(row=0, column=1)
        self.pixelYEntry = Entry(self.pixelInfoLabel, state=DISABLED, disabledforeground="black", disabledbackground="white", justify=CENTER)
        self.pixelYEntry.grid(row=1, column=1)
        self.pixelRedEntry = Entry(self.pixelInfoLabel, state=DISABLED, disabledforeground="black", disabledbackground="white", justify=CENTER)
        self.pixelRedEntry.grid(row=2, column=1)
        self.pixelGreenEntry = Entry(self.pixelInfoLabel, state=DISABLED, disabledforeground="black", disabledbackground="white", justify=CENTER)
        self.pixelGreenEntry.grid(row=3, column=1)
        self.pixelBlueEntry = Entry(self.pixelInfoLabel, state=DISABLED, disabledforeground="black", disabledbackground="white", justify=CENTER)
        self.pixelBlueEntry.grid(row=4, column=1)
        # Switch for optimization
        self.switchOptimizedState = StringVar(value="on")
        self.optimizationSwitch = ctk.CTkSwitch(self.frame, text="Optimization", variable=self.switchOptimizedState, onvalue="on", offvalue="off", button_color="black")  # progress_color="blue"
        self.optimizationSwitch.grid(row=4, column=0, sticky="WE")
        # LabelFrame for binarization
        self.binarizationOperationsLabel = LabelFrame(self.frame, text="Binarization", padx=10, pady=10, labelanchor="nw")
        self.binarizationOperationsLabel.grid(row=6, column=0, sticky="WE")
        # RadioButtons for binarization
        self.operationType = StringVar(value="3")
        self.radioManually = Radiobutton(self.binarizationOperationsLabel, text="Manually", value="1", variable=self.operationType, command=self.onOperationSelect)
        self.radioManually.grid(row=1, column=0, sticky="W", columnspan=2)
        self.radioPercentBlackSelection = Radiobutton(self.binarizationOperationsLabel, text="Percent Black Selection", value="2", variable=self.operationType, command=self.onOperationSelect)
        self.radioPercentBlackSelection.grid(row=2, column=0, sticky="W", columnspan=2)
        self.radioMeanIteration = Radiobutton(self.binarizationOperationsLabel, text="Mean iteration", value="3", variable=self.operationType, command=self.onOperationSelect)
        self.radioMeanIteration.grid(row=3, column=0, sticky="W", columnspan=2)
        # Parameters for binarization
        vcmd = (self.binarizationOperationsLabel.register(self.validateEntry))
        self.parameterOperationsLabel = LabelFrame(self.binarizationOperationsLabel, text="Threshold:", padx=10, pady=10, labelanchor="nw")
        self.parameterOperationsLabel.grid(row=8, column=0, sticky="WE")
        self.thresholdEntry = Entry(self.parameterOperationsLabel, validate='all', validatecommand=(vcmd, '%P'))
        self.thresholdEntry.grid(row=0, column=1)
        # Submit button for binarization
        self.operationSubmitButton = Button(self.binarizationOperationsLabel, text="Perform binarization", command=self.doBinarization)
        self.operationSubmitButton.grid(row=9, column=0, sticky="WE", columnspan=2)
        bold18 = font.Font(size=18, weight="bold")
        self.dilationButton = Button(self.frame, text="Dilation", command=lambda: self.doMorphology(1))
        self.dilationButton.grid(row=7, column=0, columnspan=2, sticky="WE")
        self.dilationButton['font'] = bold18
        self.erosionButton = Button(self.frame, text="Erosion", command=lambda: self.doMorphology(2))
        self.erosionButton.grid(row=8, column=0, columnspan=2, sticky="WE")
        self.erosionButton['font'] = bold18
        self.openingButton = Button(self.frame, text="Opening", command=lambda: self.doMorphology(3))
        self.openingButton.grid(row=9, column=0, columnspan=2, sticky="WE")
        self.openingButton['font'] = bold18
        self.closingButton = Button(self.frame, text="Closing", command=lambda: self.doMorphology(4))
        self.closingButton.grid(row=10, column=0, columnspan=2, sticky="WE")
        self.closingButton['font'] = bold18
        self.thinningButton = Button(self.frame, text="Thinning", command=lambda: self.doMorphology(5))
        self.thinningButton.grid(row=11, column=0, columnspan=2, sticky="WE")
        self.thinningButton['font'] = bold18
        self.thickeningButton = Button(self.frame, text="Thickening", command=lambda: self.doMorphology(6))
        self.thickeningButton.grid(row=12, column=0, columnspan=2, sticky="WE")
        self.thickeningButton['font'] = bold18
        # Parameters for morphology windows height or width
        self.morphologyParameterLabel = Label(self.frame)
        self.morphologyParameterLabel.grid(row=13, column=0)
        validationOddNumbers = (self.binarizationOperationsLabel.register(self.validateEntryOddNumber))
        self.widthLabel = Label(self.morphologyParameterLabel, text="Width:")
        self.widthLabel.grid(row=0, column=0)
        self.widthLabel['font'] = bigFont
        self.widthEntry = Entry(self.morphologyParameterLabel, validate='all', validatecommand=(validationOddNumbers, '%P'), width=3, justify=CENTER)
        self.widthEntry.grid(row=0, column=1)
        self.widthEntry.insert(0, '3')
        self.heightLabel = Label(self.morphologyParameterLabel, text="Height:")
        self.heightLabel.grid(row=1, column=0)
        self.heightLabel['font'] = bigFont
        self.heightEntry = Entry(self.morphologyParameterLabel, validate='all', validatecommand=(validationOddNumbers, '%P'), width=3, justify=CENTER)
        self.heightEntry.grid(row=1, column=1)
        self.heightEntry.insert(0, '3')

        self.imageSpace = Canvas(self.root, bg="white")
        self.imageSpace.pack(fill="both", expand=True)
        self.image = None
        self.imageId = None
        self.movedX = 0
        self.movedY = 0
        self.originalImage = None
        self.pixels = None
        self.startTime, self.endTime = None, None

    def dilation(self, widthWindow, heightWindow, optimized):
        self.measureTime("START")
        if self.image:
            height, width, _ = self.pixels.shape
            padHeight = heightWindow // 2
            padWidth = widthWindow // 2
            paddedImage = np.pad(self.pixels, ((padHeight, padHeight), (padWidth, padWidth), (0, 0)), mode='edge')
            if optimized:
                reds, greens, blues = paddedImage[:, :, 0], paddedImage[:, :, 1], paddedImage[:, :, 2]
                redSquares, greenSquares, blueSquares = np.lib.stride_tricks.sliding_window_view(reds, (heightWindow, widthWindow)), np.lib.stride_tricks.sliding_window_view(greens, (heightWindow, widthWindow)), np.lib.stride_tricks.sliding_window_view(blues, (heightWindow, widthWindow))
                dilationRed, dilationGreen, dilationBlue = np.max(redSquares, axis=(2, 3)), np.max(greenSquares, axis=(2, 3)), np.max(blueSquares, axis=(2, 3))
                self.pixels[:, :, 0][:, :], self.pixels[:, :, 1][:, :], self.pixels[:, :, 2][:, :] = dilationRed, dilationGreen, dilationBlue
            else:
                dilatedPixels = deepcopy(self.pixels)
                for y in range(0, height):
                    for x in range(0, width):
                        dilatedPixels[y, x, 0] = np.max(paddedImage[y:y+heightWindow, x:x+widthWindow, 0])
                        dilatedPixels[y, x, 1] = np.max(paddedImage[y:y + heightWindow, x:x + widthWindow, 1])
                        dilatedPixels[y, x, 2] = np.max(paddedImage[y:y + heightWindow, x:x + widthWindow, 2])
                self.pixels = deepcopy(dilatedPixels)
            self.limitPixelsAndShowImage(self.pixels, True)
        self.measureTime("END")


    def erosion(self):
        ic("2")

    def opening(self):
        ic("3")

    def closing(self):
        ic("4")

    def thinning(self):
        ic("5")

    def thickening(self):
        ic("6")

    def doMorphology(self, operationType):
        if self.image:
            # self.greyConversion()
            # self.meanIterationBinarization(True)
            width = int(self.widthEntry.get()) if self.widthEntry.get() != "" else 3
            height = int(self.heightEntry.get()) if self.heightEntry.get() != "" else 3
            optimization = False if self.switchOptimizedState.get() == "off" else True
            if operationType == 1:
                self.dilation(width, height, optimization)
            elif operationType == 2:
                self.erosion(width, height)
            elif operationType == 3:
                self.opening(width, height)
            elif operationType == 4:
                self.closing(width, height)
            elif operationType == 5:
                self.thinning(width, height)
            elif operationType == 6:
                self.thickening(width, height)
            else:
                return self.errorPopup("Nie ma takiej opcji")

    @staticmethod
    def validateEntry(P):
        if P == "" or (str.isdigit(P)):
            return True
        else:
            return False

    @staticmethod
    def validateEntryOddNumber(P):
        if P == "":
            return True
        if str.isdigit(P):
            intP = int(P)
            if intP % 2:
                return True
        return False

    def onOperationSelect(self):
        if self.operationType.get() == '1':
            self.parameterOperationsLabel.grid(row=8, column=0, sticky="WE")
            self.parameterOperationsLabel.configure(text="Threshold:")
        elif self.operationType.get() == '2':
            self.parameterOperationsLabel.grid(row=8, column=0, sticky="WE")
            self.parameterOperationsLabel.configure(text="Procent of black pixels:")
        elif self.operationType.get() == '3':
            self.parameterOperationsLabel.grid_forget()
        elif self.operationType.get() == '5':
            self.parameterOperationsLabel.grid_forget()
        else:
            raise Exception("Nie ma takiej opcji")

    def doBinarization(self):
        if self.image:
            self.greyConversion()
            optimization = False if self.switchOptimizedState.get() == "off" else True
            if self.operationType.get() == '1':
                if not self.thresholdEntry.get():
                    self.errorPopup("Error:Threshold was not given.\nYou must give threshold parameter to do that binarization")
                else:
                    threshold = int(self.thresholdEntry.get())
                    if 0 <= threshold <= 255:
                        self.thresholdBinarization(threshold, optimization)
                        # self.thresholdBinarization(threshold) if self.switchOptimizedState.get() == "off" else self.thresholdBinarizationOptimized(threshold)
                    else:
                        self.errorPopup(f"Error:Threshold must be in range of 0 to 255 not {threshold}")
            elif self.operationType.get() == '2':
                if not self.thresholdEntry.get():
                    self.errorPopup("Error:Percent of black pixels was not given.\nYou must give percent of black pixels parameter to do that binarization")
                else:
                    percent = int(self.thresholdEntry.get())
                    if 0 <= percent <= 100:
                        self.percentBlackPixelsBinarization(percent, optimization)
                        # self.percentBlackPixelsBinarization(percent) if self.switchOptimizedState.get() == "off" else self.percentBlackPixelsBinarizationOptimized(percent)
                    else:
                        self.errorPopup(f"Error:Percent of black pixels must be in range of 0 to 100 not {percent}")
            elif self.operationType.get() == '3':
                self.meanIterationBinarization(optimization)
            # elif self.operationType.get() == '5':
            #     self.minimumErrorBinarization() if self.switchOptimizedState.get() == "off" else self.minimumErrorBinarization()
            else:
                self.reloadOriginalJPG()
                print("Nie ma takiej operacji")
        else:
            self.errorPopup("Error: There's no image loaded.")

    def meanIterationBinarization(self, optimized):
        self.measureTime("START")
        if self.image:
            histogram = self.createHistogram()
            threshold = 128
            while True:
                # podział histogramu na lewa i prawa strone
                leftPart = histogram[:threshold]
                rightPart = histogram[threshold:]
                # stworzenie tablicy indexów lewej strony
                leftIndexes = np.arange(len(leftPart))
                # srednia lewej strony wyliczona z sumy (index * wartosc dla tego indexu) dzielona przez sume wartości strony
                leftMean = np.sum(leftIndexes * leftPart) / np.sum(leftPart)
                # stworzenie tablicy indexów prawej strony
                rightIndexes = np.arange(len(leftPart), len(leftPart)+len(rightPart))
                # srednia prawej strony wyliczona z sumy (index * wartosc dla tego indexu) dzielona przez sume wartości strony
                rightMean = np.sum(rightIndexes * rightPart) / np.sum(rightPart)
                # srednia z lewej i prawej strony
                newThreshold = round((leftMean + rightMean) / 2)
                # print(f"Lewa czesc:\n{leftPart}\nPrawa czesc:\n{rightPart}\nLewe indexy:\n{leftIndexes}\nPrawe indexy:\n{rightIndexes}\nLewa srednia: {leftMean}\nPrawa srednia: {rightMean}\nStary prog: {threshold}\nNowy prog: {newThreshold}")
                # sprawdzenie czy nowy prog jest rowny staremu, jesli nie to powtarzamy operacje
                if newThreshold == threshold:
                    break
                else:
                    threshold = newThreshold
            print(f"Ostateczny prog = {threshold}")
            # lookup table
            thresholdTable = np.zeros(256, dtype=np.int32)
            for i in range(threshold, 256):
                thresholdTable[i] = 255
            if optimized:
                self.pixels = thresholdTable[self.pixels]
            else:
                height, width, _ = self.pixels.shape
                for y in range(0, height):
                    for x in range(0, width):
                        for c in range(3):
                            self.pixels[y, x, c] = thresholdTable[self.pixels[y, x, c]]
            self.limitPixelsAndShowImage(self.pixels, True)
        self.measureTime("END")

    # def minimumErrorBinarization(self):
    #     print("Min error")

    def percentBlackPixelsBinarization(self, percent, optimized):
        self.measureTime("START")
        if self.image:
            histogram = self.createHistogram()
            height, width, color = self.pixels.shape
            allColor = height*width
            requirement = allColor * percent / 100
            sumValues = 0
            thresholdIndex = 0
            for index, value in enumerate(histogram):
                sumValues += value
                if sumValues >= requirement:
                    thresholdIndex = index
                    break
            # lookup table
            thresholdTable = np.zeros(256, dtype=np.int32)
            for i in range(256):
                thresholdTable[i] = 255 if i >= thresholdIndex else 0
            if optimized:
                self.pixels = thresholdTable[self.pixels]
            else:
                height, width, _ = self.pixels.shape
                for y in range(0, height):
                    for x in range(0, width):
                        for c in range(3):
                            self.pixels[y, x, c] = thresholdTable[self.pixels[y, x, c]]
            self.limitPixelsAndShowImage(self.pixels, True)
        self.measureTime("END")

    def createHistogram(self):
        self.measureTime("START")
        if self.image:
            histogram = np.zeros(256, dtype=np.int32)
            # height, width, color = self.pixels.shape
            # for y in range(height):
            #     for x in range(width):
            #         self.histogram[self.pixels[y, x, 0]] += 1
            uniqueValues, counts = np.unique(self.pixels[:, :, 0], return_counts=True)
            #print(uniqueValues, counts)
            for value, count in zip(uniqueValues, counts):
                histogram[value] = count
            # print(self.histogram)
        self.measureTime("END")
        return histogram

    def thresholdBinarization(self, thresholdValue, optimized):
        self.measureTime("START")
        if self.image:
            # # lookup table
            thresholdTable = np.zeros(256, dtype=np.int32)
            for i in range(256):
                thresholdTable[i] = 255 if i >= thresholdValue else 0
            if optimized:
                self.pixels = thresholdTable[self.pixels]  # version with use of lookup table
                # self.pixels = np.where(self.pixels >= thresholdValue, 255, 0) #  simple version
            else:
                height, width, _ = self.pixels.shape
                for y in range(0, height):
                    for x in range(0, width):
                        for c in range(3):
                            self.pixels[y, x, c] = thresholdTable[self.pixels[y, x, c]]
                            #self.pixels[y, x, c] = 255 if self.pixels[y, x, c] >= thresholdValue else 0
            self.limitPixelsAndShowImage(self.pixels, True)
        self.measureTime("END")

    def greyConversion(self, adjusted=True):
        self.measureTime("START")
        if self.image:
            # Zrobienie sredniej z wyswietlanych pixeli na ekranie
            if adjusted:
                averages = 0.299 * self.pixels[:, :, 0] + 0.587 * self.pixels[:, :, 1] + 0.114 * self.pixels[:, :, 2]
                self.pixels[:, :, 0] = averages
                self.pixels[:, :, 1] = averages
                self.pixels[:, :, 2] = averages
            else:
                averages = (self.pixels[:, :, 0] + self.pixels[:, :, 1] + self.pixels[:, :, 2]) / 3
                self.pixels[:, :, 0] = averages
                self.pixels[:, :, 1] = averages
                self.pixels[:, :, 2] = averages
        self.measureTime("END")

    def errorPopup(self, information=None):
        self.errorLabel = Label(Toplevel(), text=information, padx=20, pady=20)
        self.errorLabel.pack(side="top", fill="both", expand=True)

    def limitPixelsAndShowImage(self, pixels=None, limitPixels=False):
        if pixels is not None:
            limitedPixels = np.clip(pixels, 0, 255)
        else:
            limitedPixels = np.clip(self.pixels, 0, 255)
        if limitPixels:
            self.pixels = limitedPixels
        self.image = Image.fromarray(limitedPixels.astype(np.uint8))
        self.tkImage = ImageTk.PhotoImage(self.image)
        self.show_image()

    def updatePixelInfoLabel(self, x, y, pixel_rgb):
        if pixel_rgb is not None:
            r, g, b = pixel_rgb
            for entry, value in zip(
                    [self.pixelXEntry, self.pixelYEntry, self.pixelRedEntry, self.pixelGreenEntry, self.pixelBlueEntry], [x, y, r, g, b]):
                entry.config(state="normal")
                entry.delete(0, 'end')
                entry.insert(0, str(value))
                entry.config(state="disabled")
        else:
            for entry in self.pixelXEntry, self.pixelYEntry, self.pixelRedEntry, self.pixelGreenEntry, self.pixelBlueEntry:
                entry.config(state="normal")
                entry.delete(0, 'end')
                entry.config(state="disabled")

    def settingsAfterLoad(self):
        if self.imageId is not None:
            self.imageSpace.delete(self.imageId)
            self.movedX, self.movedY = 0, 0
        self.imageId = self.imageSpace.create_image(self.movedX, self.movedY, anchor="nw", image=self.tkImage)
        self.imageSpace.bind("<Motion>", self.on_mouse_move)
        self.imageSpace.bind("<Enter>", self.changeCursor)
        self.imageSpace.bind("<Leave>", self.changeCursorBack)
        self.bind_keyboard_events()
        self.bind_mouse_drag_events()
        self.zoom_settings()

    def show_image(self):
        if self.imageId:
            self.imageSpace.delete(self.imageId)
            self.imageId = None
            self.imageSpace.imagetk = None
        width, height = self.image.size
        new_size = int(self.imscale * width), int(self.imscale * height)
        imagetk = ImageTk.PhotoImage(self.image.resize(new_size))
        self.imageId = self.imageSpace.create_image(self.movedX, self.movedY, anchor='nw', image=imagetk)
        self.imageSpace.lower(self.imageId)
        self.imageSpace.imagetk = imagetk

    def loadJPG(self):
        filePath = askopenfilename()
        if filePath == '':
            return
        self.image = Image.open(filePath)
        if self.image is None:
            return
        self.pixels = np.array(self.image, dtype=np.int32)
        self.tkImage = ImageTk.PhotoImage(self.image)
        self.settingsAfterLoad()
        self.greyConversion()
        self.meanIterationBinarization(True)
        self.originalImage = deepcopy(self.image)

    def reloadOriginalJPG(self):
        if self.originalImage:
            self.image = deepcopy(self.originalImage)
            if self.image is None:
                return
            self.pixels = np.array(self.image, dtype=np.int32)
            self.tkImage = ImageTk.PhotoImage(self.image)
            self.settingsAfterLoad()

    def saveJPG(self):
        if self.image:
            file_path = asksaveasfilename(initialfile='Untitled.jpg', defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg")])
            if file_path:
                self.image.save(file_path, "JPEG")
                print(f"Image saved as {file_path}")

    # przesuwanie obrazków myszką
    def start_drag(self, event):
        self.last_x = event.x
        self.last_y = event.y

    def drag_image(self, event):
        if hasattr(self, 'last_x') and hasattr(self, 'last_y'):
            dx = event.x - self.last_x
            dy = event.y - self.last_y
            self.move_image(event, dx, dy, False)
            self.last_x = event.x
            self.last_y = event.y

    def stop_drag(self, event):
        if hasattr(self, 'last_x') and hasattr(self, 'last_y'):
            del self.last_x
            del self.last_y

    def bind_mouse_drag_events(self):
        self.imageSpace.bind("<ButtonPress-1>", self.start_drag)
        self.imageSpace.bind("<B1-Motion>", self.drag_image)
        self.imageSpace.bind("<ButtonRelease-1>", self.stop_drag)
    # zmiana kursora
    def changeCursor(self, event):
        self.imageSpace.config(cursor="cross_reverse")  # best option "pirate" XD

    def changeCursorBack(self, event):
        self.imageSpace.config(cursor="")

    def zoom_settings(self):
        self.root.bind("<MouseWheel>", self.wheel)
        self.imscale = 1.0
        self.delta = 0.75
        self.text = self.imageSpace.create_text(0, 0, anchor='nw', text='')
        self.show_image()
        self.imageSpace.configure(scrollregion=self.imageSpace.bbox('all'))

    def wheel(self, event):
        scale = 1.0
        if event.delta == -120:
            scale *= self.delta
            self.imscale *= self.delta
        if event.delta == 120:
            scale /= self.delta
            self.imscale /= self.delta
        # Rescale all canvas objects
        x = self.imageSpace.canvasx(event.x)
        y = self.imageSpace.canvasy(event.y)
        self.imageSpace.scale(self.imageId, x, y, scale, scale)
        self.show_image()

    def move_image(self, event, dx, dy, scaleMoving=False):
        if self.imageId is not None:
            if scaleMoving:
                dx *= self.imscale*2
                dy *= self.imscale*2
            self.movedX += dx
            self.movedY += dy
            self.imageSpace.move(self.imageId, dx, dy)

    def bind_keyboard_events(self):
        self.root.bind("<Left>", lambda event: self.move_image(event, dx=10, dy=0, scaleMoving=True))
        self.root.bind("<Right>", lambda event: self.move_image(event, dx=-10, dy=0, scaleMoving=True))
        self.root.bind("<Up>", lambda event: self.move_image(event, dx=0, dy=10, scaleMoving=True))
        self.root.bind("<Down>", lambda event: self.move_image(event, dx=0, dy=-10, scaleMoving=True))

    def on_mouse_move(self, event):
        # image_coords = self.imageSpace.coords(self.imageId)
        # print(f"{image_coords} {self.image.width} {self.image.height}")
        # print(f"f{self.image}")
        if self.image is not None:
            x, y = event.x-self.movedX, event.y-self.movedY
            # print(f"x={event.x} mX={self.movedX}  y={event.y} mY={self.movedY} IX={self.image.width}  IY={self.image.height}")
            # image_x, image_y = self.imageSpace.coords(self.imageId)
            # print(f"Ob = {image_x} {image_y}")
            if (0 <= x < self.image.width * self.imscale) and (0 <= y < self.image.height * self.imscale):
                pixel_rgb = self.get_pixel_color(int(x/self.imscale), int(y/self.imscale))
                self.updatePixelInfoLabel(int(x / self.imscale), int(y / self.imscale), pixel_rgb)
            elif self.pixelXEntry.get():
                self.updatePixelInfoLabel(None, None, None)

    def get_pixel_color(self, x, y):
        if self.image is not None:
            try:
                pixel = self.image.getpixel((x, y))
                return pixel
            except Exception as e:
                print(f"Error getting pixel color: {e}")
        return None

    def measureTime(self, startEnd):
        if startEnd.lower() == "start":
            self.startTime = time.perf_counter()
        elif startEnd.lower() == "end":
            self.endTime = time.perf_counter()
            execution_time = self.endTime - self.startTime
            print(f"Czas wykonania funkcji: {execution_time} sekundy")