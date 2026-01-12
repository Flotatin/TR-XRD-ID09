### CODE POUR COMMUNIQUER AVEC L'OSCILLOSCOPE LECROY, AFFICHER LES TRACES ET LES SAUVEGARDER, VERSION PyQt###
import sys
import pandas as pd
import lecroyscope
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QHBoxLayout, QWidget, QFileDialog , QLineEdit , QListWidget, QCheckBox, QLabel ,QGroupBox, QVBoxLayout
from PyQt5.QtCore import Qt
from tkinter import filedialog
import os

class OscilloscopeViewer(QMainWindow):
    def __init__(self,folder=r"F:\Aquisition_Banc_CEDd\Aquisition_LECROY_Banc_CEDd"):
        super().__init__()

        self.folder=folder
        self.setWindowTitle("Oscilloscope Data Viewer")
        self.setGeometry(100, 100, 2600, 1000)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QHBoxLayout(self.central_widget)

       

        # Création du groupe pour les autres composants
        
        layout_c1 = QVBoxLayout()
        

        #group_graphe.setMaximumWidth(300)

        # Créer un cadre pour le graphique
        self.figure, self.ax = plt.subplots(figsize=(15,15))
        self.ax.grid(True)
        self.figure.tight_layout()
        self.canvas = FigureCanvas(self.figure)
        layout_c1.addWidget(self.canvas)
        
        self.toolbar = NavigationToolbar(self.canvas,self)
        layout_c1.addWidget(self.toolbar)


        layout_check = QVBoxLayout()
        

        self.text= QLabel("",self)
        layout_c1.addWidget(self.text)

        layout_trace = QVBoxLayout()
        self.fichier_trace_listbox=QListWidget(self)
        self.fichier_trace_listbox.itemClicked.connect(self.load_trace)
        layout_trace.addWidget(self.fichier_trace_listbox)
        if self.folder:
            files = os.listdir(self.folder)  # Obtenir la liste des fichiers dans le dossier
            self.fichier_trace_listbox.clear()
            self.fichier_trace_listbox.addItems(files)

        self.file_name_entry = QLineEdit()
        self.file_name_entry.returnPressed.connect(self.search_trace)
        layout_trace.addWidget( self.file_name_entry)

        self.search_button = QPushButton("Rechercher")
        self.search_button.clicked.connect(self.search_trace)
        layout_trace.addWidget(self.search_button)

        
        self.liste_check_load =[ QCheckBox(f"Channel{i} Scope", self) for i in range(1,5)] + [ QCheckBox(f"Channel{i} Load", self) for i in range(1,5)]
        for check in self.liste_check_load:
            check.setChecked(True)
            check.clicked.connect(self.print_plot)
            layout_check.addWidget(check)

        # Bouton pour acquérir et afficher les traces
        self.acquire_button = QPushButton("Acquire and Display", self)
        self.acquire_button.clicked.connect(self.acquire_and_display)
        layout_check.addWidget(self.acquire_button)

        self.name_save_entry= QLineEdit(self)
        layout_check.addWidget(self.name_save_entry)
        # Bouton pour enregistrer les données
        self.save_button = QPushButton("Save Data", self)
        self.save_button.clicked.connect(self.save_data)
        layout_check.addWidget(self.save_button)

        # Bouton pour acquérir et afficher les traces
        self.load_button = QPushButton("Load Trace", self)
        self.load_button.clicked.connect(self.load_trace_button)
        layout_check.addWidget(self.load_button)

        self.clear_button=QPushButton("CLEAR", self)
        self.clear_button.clicked.connect(self.clear_plot)
        layout_check.addWidget(self.clear_button)


        self.id_entry= QLineEdit(self)
        self.id_entry.setText("100.100.143.2")
        layout_check.addWidget(self.id_entry)
        self.scope_button=QPushButton("Connect scope", self)
        self.scope_button.clicked.connect(self.connect_scope)
        layout_check.addWidget(self.scope_button)
       
        group_trace = QGroupBox("Trace")
        group_trace.setLayout(layout_trace)
        self.layout.addWidget(group_trace)


        group_graphe = QGroupBox("Plot")
        group_graphe.setLayout(layout_c1)
        self.layout.addWidget(group_graphe)

        group_check = QGroupBox("Print or not")
        group_check.setLayout(layout_check)
        self.layout.addWidget(group_check)

         # Création du groupe pour les autres composants


        self.text_scope= QLabel("Scope not connect",self)
        layout_c1.addWidget(self.text)
        layout_c1.addWidget(self.text_scope)

        

        self.plot_load=[]
        self.plot_oscilo=[]
        self.plot_cursor1=[self.ax.axvline(0, color='k', lw=1.3, ls='-.', alpha=0.7 ),self.ax.axhline(0, color='k', lw=1.3, ls='-.', alpha=0.7 )]
        self.plot_cursor2=[self.ax.axvline(0, color='k', lw=1.3, ls='--', alpha=0.7 ),self.ax.axhline(0, color='k', lw=1.3, ls='--', alpha=0.7 )]
        self.cursor_value=[0,0,0,0,0,0]
        self
        self.text_cursor="X1: {} X2: {} DX: {} \nY1: {} Y2: {} DY: {}".format(0,0,0,0,0,0)
        
        self.text_code="Welcome"

        self.text_help="cursor 1 press 'a' cursor2 press 'b'"

        self.text.setText(self.text_cursor + "\n"+ self.text_code  + "\n"+ self.text_help)
        self.ax.figure.canvas.mpl_connect('button_press_event', self.clic_cursor)
        self.connect_scope()
        self.bit_bypass=False
        self.touche=None

    def connect_scope(self):
        try:
            self.scope = lecroyscope.Scope( self.id_entry.text())  # IP address of the scope "169.254.13.6"
            self.text_scope.setText(f"Scope connect ID: {self.scope.id}")

        except Exception as e:
            self.text_code ="ERROR:"+str(e)
            self.text_scope.setText(f"Scope fail")
        
        self.text.setText(self.text_cursor + "\n"+ self.text_code  + "\n"+ self.text_help)
    

    def search_trace(self):# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - SEARCH JCPDS - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        file_name = self.file_name_entry.text()

        self.fichier_trace_listbox.clear()
        for file in os.listdir(self.folder):
            if file_name in file:
                self.fichier_trace_listbox.addItem(file)


    """
    def keyPressEvent(self,event):
        if self.canvas.hasFocus():
            key = event.key()
            if key == Qt.Key_A :
                self.touche="c1"
                self.text_code="Select X1 and Y1"
                self.text.setText(self.text_cursor + "\n"+ self.text_code  + "\n"+ self.text_help)
            elif key == Qt.Key_B :
                self.touche="c2"
                self.text_code="Select X2 and Y2"
                self.text.setText(self.text_cursor + "\n"+ self.text_code  + "\n"+ self.text_help)

    def keyReleaseEvent(self):
        if self.canvas.hasFocus():
            self.touche=None
            self.text_code="Welcome"
            self.text.setText(self.text_cursor + "\n"+ self.text_code  + "\n"+ self.text_help)
    """
    def clic_cursor(self,event):
        self.setFocus()
        if event.inaxes ==self.ax and (self.touche =="c1" or self.touche =="c2"):
            x, y = event.xdata, event.ydata
            if self.touche =="c1":
                c=0
                self.plot_cursor1[0].set_xdata([x])
                self.plot_cursor1[1].set_ydata([y])
            elif self.touche =="c2":
                c=1
                self.plot_cursor2[0].set_xdata([x])
                self.plot_cursor2[1].set_ydata([y])

            self.cursor_value[0+c]=round(x,3)
            self.cursor_value[3+c]=round(y,3)
            self.cursor_value[2]=round(self.cursor_value[1]-self.cursor_value[0],3)
            self.cursor_value[5]=round(self.cursor_value[4]-self.cursor_value[3],3)
            self.text_cursor="X1: {} X2: {} D21X: {} \nY1: {} Y2: {} D21Y: {}".format(self.cursor_value[0],self.cursor_value[1],self.cursor_value[2],self.cursor_value[3],self.cursor_value[4],self.cursor_value[5])
            self.text.setText(self.text_cursor + "\n"+ self.text_code  + "\n"+ self.text_help)
            self.canvas.draw()


    def print_plot(self):
        for i in range(4):
            if i < len(self.plot_oscilo):
                self.plot_oscilo[i].set_visible(self.liste_check_load[i].isChecked())

            if i < len(self.plot_load):
                self.plot_load[i].set_visible(self.liste_check_load[i+4].isChecked())
        self.canvas.draw()


    def clear_plot(self):
        self.ax.clear()
        self.canvas.draw()

    def load_trace(self,item):
        c_t=["k","darkorange","darkred","darkblue","darkgreen"]
        #file =filedialog.askopenfilename(title="Sélectionner TRACE")
        #if file:
        chemin_fichier = os.path.join(self.folder, item.text())
        #self.liste_objets_widget.setCurrentRow(item)
        index = self.fichier_trace_listbox.row(item)
        
        if index  :
            try:
                oscilo=pd.read_csv(chemin_fichier, sep='\s+', skipfooter=0, engine='python')
                if self.plot_load != []:
                    for p in self.plot_load:
                        if p is not None:
                            p.remove()
                self.plot_load =[]
                for i in range(1,5):
                    self.plot_load.append(self.ax.plot(oscilo["Time"]*1e3,oscilo[f"Channel{i}"],'.-',label=f"Channel{i}",c=c_t[i])[0])
                self.canvas.draw()
            except Exception as e:
                print(e)
                self.text_code="Error in loading"
        
        self.text.setText(self.text_cursor + "\n"+ self.text_code  + "\n"+ self.text_help)
    
    def load_trace_button(self):
        c_t=["k","darkorange","darkred","darkblue","darkgreen"]
        file =filedialog.askopenfilename(title="Sélectionner TRACE")
        if file:
            oscilo=pd.read_csv(file, sep='\s+', skipfooter=0, engine='python')
            if self.plot_load != []:
                for p in self.plot_load:
                    if p is not None:
                        p.remove(file)
            self.plot_load =[]
            for i in range(1,5):
                self.plot_load.append(self.ax.plot(oscilo["Time"]*1e3,oscilo[f"Channel{i}"],'.-',label=f"Channel{i}",c=c_t[i])[0])
            
            self.canvas.draw()

        self.text.setText(self.text_cursor + "\n" +self.text_code)


    def acquire_and_display(self):
        # Code pour acquérir les traces (à partir de votre code existant)
        trace_group = self.scope.read(1, 2, 3, 4)
        time = trace_group.time  # time values are the same for all traces

        # Effacer le contenu précédent du graphique
        if self.plot_oscilo != []:
            for p in self.plot_oscilo:
                if p is not None:
                    p.remove()
        self.plot_oscilo =[]
        # Afficher les nouvelles traces sur le graphique
        c_t=["k","gold","r","b","g"]
        for i in range(1, len(trace_group) + 1):
            self.plot_oscilo.append(self.ax.plot(trace_group[i].x*1e3, trace_group[i].y, '.-',c=c_t[i])[0])
            self.ax.set_xlabel('Time (ms)')
            self.ax.set_ylabel(f'Channel {i}')
            self.ax.set_title(f'Trace for Channel {i}')
            self.ax.grid(True)
            self.figure.tight_layout()
            self.canvas.draw()

        self.text.setText(self.text_cursor + "\n" +self.text_code)
    def save_data(self):
        trace_group = self.scope.read(1, 2, 3, 4)
        time = trace_group.time  # time values are the same for all traces
        df = pd.DataFrame({"Time" :pd.Series(time), 
                   "Channel1" :pd.Series(trace_group[1].y),
                   "Channel2" :pd.Series(trace_group[2].y),
                   "Channel3" :pd.Series(trace_group[3].y),
                   "Channel4" :pd.Series(trace_group[4].y),
                  })
        #file_path, _ = QFileDialog.getSaveFileName(self, "Save Data", "", "Data Files (*.dat)")
        file_path =os.path.join(self.folder,str(self.name_save_entry.text()))
        if file_path:
            with open(file_path, 'w') as file2write:
                file2write.write(df.to_string())
            print(f"Data saved to {file_path}")
            self.text_code=f"Data saved to {file_path}"
        self.text.setText(self.text_cursor + "\n"+ self.text_code  + "\n"+ self.text_help)

    def select_folder(self):
            # Fonction pour parcourir un dossier et afficher ses fichiers
        options = QFileDialog.Options()
        self.folder = QFileDialog.getExistingDirectory(self, "Sélectionner un dossier", options=options)
        if self.folder:
            files = os.listdir(self.folder)  # Obtenir la liste des fichiers dans le dossier
            self.fichier_trace_listbox.clear()
            self.fichier_trace_listbox.addItems(files)
            self.text_code=f"Folder select {self.folder}"
        self.text.setText(self.text_cursor + "\n"+ self.text_code  + "\n"+ self.text_help)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = OscilloscopeViewer()
    viewer.show()
    sys.exit(app.exec_())