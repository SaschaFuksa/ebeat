# ebeat project
![](ebeat.png)
Projektmanagement: [Trello Board](https://trello.com/b/HYG7tuaq/tech-lab) 

<p>Patrick Treiber 42609</p>
<p>Michael Henn 42595 </p>
<p>Vanessa Hartmann 42580 </p>
<p>Nicola Haller 42617 </p>
<p>Sascha Fuksa 42590 </p>
<p>Arline Carle 42582 </p>

##  Anforderunsganalyse 
###  Ziel der Anforderung: 
Das Ziel der zu entwickelnden Maschine ist, dass sie automatisiert Musik zusammenstellt.

###  Anwendungsbereich:  

Der Anwendungsbereich einer zu entwickelnden KI ist, dass die KI selbstständig (automatisiert) Musikstücke zerlegt und wieder zu neuen Musikstücken zusammenstellt.  


### Use Case    

<U> ebeat – Samplecreator </U>
![](./use-cases/ebeat-samplecreator.png)

**Name Use Case:** Choose input song<br>
**Akteur:** Lisa<br>
**Ziel:** Lisa kann einen bestimmten Song auswählen.<br>
**Vorbedingung:** Es muss eine Audiodatei (wav/mp3) in einem Pfad vorliegen.<br> 
**Nachbedingung:** Der Pfad wurde ausgewählt, die Datei liegt zum Einspielen bereit.<br> 
**Standardablauf:** <br>1. Lisa öffnet die Nutzeroberfläche des Systems.<br> 2. Lisa gibt den Befehl in die Nutzeroberfläche ein, der es ermöglicht einen Inputpfad anzugeben.<br> 3. Lisa trägt den Input Pfad ein und drück Eingabe (Enter).  

**Name Use Case:** Cut tracks in sample<br>
**Akteur:** Lisa<br>
**Ziel:** Erstellung von Samples aus einem Musikstück.<br> 
**Vorbedingung:** Es müssen einer oder mehrere Songs im ausgewählten Input Pfad hinterlegt sein. Es muss bestimmt sein, welche Art von Cutting angewendet werden soll und welche Parameter gewünscht sind (Samplegröße oder Sampleanzahl).<br>
**Nachbedingung:** Musikstücke wurden in einzelne Samples zerlegt.<br> 
**Standardablauf:**<br> 1. Lisa öffnet die Nutzeroberfläche des Systems.<br> 2. Lisa wählt die Songs aus, die zerlegt werden sollen.<br> 3. Lisa gibt an, mit welcher Methodik zerlegt werden soll.<br> 4. Lisa startet den Vorgang.<br>

**Name Use Case:** Save samples<br> 
**Akteur:** Lisa<br>
**Ziel:** Lisa möchte die Samples an einem bestimmten Pfad ablegen.<br>
**Vorbedingung:** Output Pfad muss vorhanden sein.<br>
**Nachbedingung:** Musik Samples liegen im Output Pfad bereit.<br>
**Standardablauf:**<br> 1. Lisa öffnet die Nutzeroberfläche des Systems.<br> 2. Lisa gibt den Befehl in die Nutzeroberfläche ein, der es ermöglicht einen Output Pfad anzugeben.<br> 3. Lisa trägt den Output Pfad ein und drück Eingabe (Enter). 
<br>
<br>

 #### <U> ebeat – ML Trainer </U>
![](./use-cases/ebeat_mltrainer.png)

**Name Use Case:** Check concatenation matching of samples<br>
**Akteur:** Lisa <br>
**Ziel:** Harmonie zwischen den einzelnen Samples überprüfen. <br>
**Vorbedingung:**  Samples müssen aus dem Output Pfad (Save Samples) gelesen werden können.<br> 
**Nachbedingung:** Musikstücke wurden auf Harmonisierung geprüft. <br>
**Standardablauf:**<br> 1. Lisa gibt den Output Pfad der gespeicherten Samples an.<br>  2. Lisa startet den ML- Algorithmus. <br>

**Name Use Case:** Choose trainingsdata <br>
**Akteur:** Lisa <br>
**Ziel:** Trainingsdaten werden für den ML-Algorithmus festgelegt.<br> 
**Vorbedingung:** Samples für das Training müssen im Output Pfad vorliegen. <br>
**Nachbedingung:** Trainingsdaten Samples ausgewählt. <br> 
**Standardablauf:**<br> 1. Die Sample Dateien werden im Output Pfad (Samples) ausgewählt.<br> 

**Name Use Case:** Train AI <br>
**Akteur:** Lisa <br>
**Ziel:** Die ML-Modell durchläuft den Lernalgorithmus, erstellt und speichert ein Modell. <br>
**Vorbedingung:**  Samples für das Training müssen im Output Pfad (Samples) ausgewählt sein. <br>
**Nachbedingung:** Modell erstellt, kann für weitere Input Samples wiederverwendet werden.<br>
**Standardablauf:** <br>1. ML-Modell zieht die Samples Dateien aus dem Output Pfad.<br> 2. ML-Modell trainiert anhand des Lernalgorithmus. 3. Modell wird erstellt und gespeichert. 

**Name Use Case:** Evaluate AI result <br>
**Akteur:** Lisa <br>
**Ziel:** Bewertung des erstellten Modells aus dem Schritt Train AI. <br>
**Vorbedingung:**  Das Modell wurde trainiert und erstellt (gespeichert). <br>
**Nachbedingung:**  Das Modell wurde evaluiert. <br>
**Standardablauf:** <br>1. Lisa prüft die Temporäre Musikdatei auf Hörbarkeit.<br> 2. Lisa bewertet das Modell und startet bei Nichtgefallen einen erneuten Trainingsdurchlauf (Train AI). 
<br>
<br>


#### <U> ebeat-Resampler </U>
![](./use-cases/ebeat-resampler.png)

**Name Use Case:** Choose input samples <br>
**Akteur:** Lisa <br>
**Ziel:** Auswählen mehrerer Samples, um diese harmonisiert zusammenzuführen. <br>
**Vorbedingung:**  Es müssen mehrere Audiosamples in einem Pfad vorliegen. <br>
**Nachbedingung:** Die fertige Audiodatei / Stream liegt im ausgewählten Directory zum Abspielen bereit. <br>
**Standardablauf:** <br>1. Lisa öffnet den Resampler.<br> 2. Lisa gibt den Befehl in die Nutzeroberfläche ein, der es ermöglicht einen Input Pfad anzugeben.<br> 3. Lisa trägt den Input Pfad ein und drückt Eingabe (Enter).<br> 4. Lisa startet den Resampler und erstellt eine Audiodatei / Stream.<br> 


**Name Use Case:** Generate music file <br>
**Akteur:** Lisa<br> 
**Ziel:** System kreiert anhand der zusammengeführten Samples eine Musik Datei.<br> 
**Vorbedingung:**  Es müssen mehrere Samples im ausgewählten Input directory hinterlegt sein, damit diese nacheinander verkettet werden können. Das Trainingsmodell muss erstellt sein. <br>
**Nachbedingung:** Aus den Samples wurde eine Musikdatei / Stream erstellt. <br>
**Standardablauf:**<br> 1. Lisa öffnet den Resampler und lässt das ML-Modell die vorher ausgewählten Samples zu einem Musikstück komponieren/ aneinanderreihen.<br> 


**Name Use Case:** Save music file <br>
**Akteur:** Lisa<br>
**Ziel:** Lisa legt das Musikstück in einem bestimmten Pfad ab. <br>
**Vorbedingung:** Output Directory muss vorhanden sein. Audiodatei / Stream muss vorhanden sein. <br>
**Nachbedingung:** Generiertes Musikstück wurde gespeichert. <br>
**Standardablauf:**<br> 1. Lisa öffnet den Resampler. <br>2. Lisa gibt den Befehl in die Nutzeroberfläche ein, der es ermöglicht einen Output Pfad anzugeben.<br>3. Lisa trägt den Output Pfad ein und drückt Eingabe (Enter). <br>4. Daraufhin wird das Musikstück gespeichert und ist zum Abhören bereit. <br>
##  Systemspezifikation
### Komponentendiagramm  
![img.png](./diagrams/ComponentDiagram.png)
Das Komponenten Diagramm dient zur übersichtlichen Darstellung der einzelnen Teilkomponenten des Gesamtsystems. Die Audiokomponente ist zuständig für die Verarbeitung von Musikformaten. Sie kann diese laden, speichern und verarbeiten. Der Compiler, welche über eine Schnittstelle mit der Audiokomponente verbunden ist, wandelt die Audiodateien in ein geeignetes Zahlenformat um. Die letzte Komponente unseres Gesamtsystem, die ML-Komponente, ist in der Lage ein Modell zu trainieren, welches Samples bewertet und Feedback an die Audiokomponente hinsichtlich ihrer Eignung der Konkatenation. 
### Klassendiagramm  
#### <U> Audio Component </U>
![img_1.png](./diagrams/AudioComponent.png)
Das Klassendiagramm ”Audio Component” bildet die Klassen MusicSampleCreator, MusicFileCollector, SampleSaver sowie die abstrakte Klasse MusicSampleCutter und ihre beiden erbenden Klassen StochasticMusicSampleCutter und EqualMusicSampleCutter ab. In der Klasse MusicSampleCreator werden die Samples erstellt, dafür ruft sie die jeweiligen Methoden der weiteren Klassen auf. Über die Klasse MusicFileCollector werden die Inputtracks geladen. Die Klasse SampleSaver speichert die Samples und die Metadaten. Die beiden erbenden Klassen der abstrakten Klasse MusicSampleCutter schneiden die Tracks in Samples. Die Zerschneidung der Tracks in Samples erfolgt entweder äquidistant oder nach stochastischem Ansatz.  
- Laden der Audio-Files  
- Audio-Files müssen sinnvoll geteilt werden (Samples), manuell oder stochastisch 
- Nummerierung/Benennung für Zuordnung der Audio-Files notwendig 
- Samples müssen zu einem Stream/Audio-File zusammengesetzt werden können 
- Speicherung des Audio Streams/Audio-Files 
- Kann Audio-Files (WAV/MP3 -> Format egal) verwenden 
- Audioeigenschaften können analysiert werden -> Metadaten 
- Erzeugung eines Streams/Audio-Files (Reihenfolge) anhand einzelner Samples  
- Speichern der Audiofiles 
#### <U> ML-Component </U>
![img_2.png](./diagrams/MLComponent.png)
Das ML-Komponente beinhaltet Klassen, die ein neuronales Netzwerk abbilden. Dazu gehören die Klassen NeuralLayer und Neuron. Die Klasse EbeatNeuralNetwork trainiert das neuronale Netz und entscheidet welche Samples zueinander passen. Trainiert wird das Trainingsmodell, das am Ende Grundlage für die Entscheidungsfindung ist. 
- Samples werden in mathematischer Form = Zahlenformat konvertiert (Tabelle/Matrix) 
- Übersetzt Samples anhand eines geeigneten Formates -> Tensor für ML-Komponente 
- Leitet die übersetzten Formate an ML-Komponente weiter 
#### <U> Complier Component </U>
![img_3.png](./diagrams/CompilerComponent.png)
Die Compiler-Komponente dient zu Übersetzung der Musikformate. Darin sind die Klassen SampleLoader, welcher die Samples aus einem bestimmten Pfad lädt. Die Klasse SampleCompiler nimmt die Übersetzung in ein nummerisches Format vor. 
- Lernalogrythmus identifizieren und bewertet ob einzelne Samples harmonieren -> Notwendig für das Erstellen des Streams/Audio-Files, dies muss von Audio-Komponente rückgemeldet werden um die einzelnen Samples zusammenzuführen 
- RNN / LSTM wird anhand Tensorflow Machine Learning Plattform umgesetzt 
- Audioeigenschaften (Amplitude -> Lautstärke, Wellenlänge -> Schnelligkeit, Phase -> Verschiebung) werden sinnvoll betrachtet und ggf. verwendet 
- Speicherung der neuen Musikstücke 
## Theoretische Grundlagen



