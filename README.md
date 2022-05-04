# ebeat project
![](Pictures/ebeat.png)
Projektmanagement: [Trello Board](https://trello.com/b/HYG7tuaq/tech-lab) 

Patrick Treiber 42609<br>
Michael Henn 42595<br>
Vanessa Hartmann 42580<br>
Nicola Haller 42617<br>
Sascha Fuksa 42590<br>
Arline Carle 42582<br>

##  Anforderunsganalyse 
###  Ziel der Anforderung: 
Das Ziel der zu entwickelnden Maschine ist, dass sie automatisiert Musik zusammenstellt.

###Anforderungen

A1: Das System soll das Erlernen von zueinander passenden Musik-Samples
ermöglichen.

A2: Es sind Musikstücke in geeignetem Format bereitzustellen.

A3: Die Musikstücke müssen durch das Zielsystem nach einer wählbaren
Strategie zerstückelt werden können. Dabei sind geeignete Namen für die
Samples zu verwenden. Ggf. sind weitere Metadaten zu verwalten.

A4: Die Samples werden hinsichtlich ihrer Audioeigenschaften geeignet
analysiert. Der Audiostream muss als Zahlenstream aufgefasst werden können.

A5: Eine Inferenz1-Komponente leitet aus den Samples geeignete Musikstücke
ab, speichert diese und ermöglicht so die Erstellung neuer Tracks.

A6: Für die Dokumentation von Software-Strukturen ist die Unified Modeling
Language (UML) oder die deklarative Beschreibung mit User Stories zu
verwenden.

A7: Für die Inferenz im Allgemeinen ist Tensorflow zu verwenden.

A8: Verwendete Programmiersprachen: Sofern eine PythonAPI der verwendeten
Dienste und Komponenten verfügbar ist, sind diese zu nutzen. Hierzu muss
innerhalb der Gruppe ein Konsens erzielt werden.

A9: Falls erforderlich sind für die Integration (Nutzung und Bereitstellung)
von Diensten Web Services nach dem REST-Ansatz zu nutzen.

A10: Es wird den Teams überlassen, inwiefern Ansätze wie Reinforcement
Learning einsetzen.

A11: Für das Lernen sind Ansätze wie LSTM, Recurrent Neural Networks (RNN),
Attention etc.in Betracht zu ziehen.

###  Anwendungsbereich:

Der Anwendungsbereich einer zu entwickelnden KI ist, dass die KI selbstständig (automatisiert) Musikstücke zerlegt und wieder zu neuen Musikstücken zusammenstellt.  


### Use Case    

<U> ebeat – Samplecreator </U>

![](use-cases/ebeat-samplecreator.png)

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
Mit Hilfe eines neuronalen Netzwerks, ist es möglich einer Maschine etwas
anzutrainieren. Dieses besteht aus zwei wesentlichen Bestandteilen:

-   Neuron: Ein Knoten mit Gewichtung, Bias und Aktivierungsfunktion, mit der
    berechnet wird, ob das Neuron “feuert”. Beispiele für Aktivierungsfunktionen
    sind RELU, Sigmoid und Tanh

-   Layer: Eine Sammlung von Neuron. Dabei unterscheidet man zwischen Input-,
    Output, und Hiddenlayer. Verfügt das Netzwerk über mehr als ein Hiddenlayer,
    spricht man von deep learning.

Das Erstellen eines neuronalen Netzwerks kann beispielweise mit TensorFlow und
Keras erfolgen. TensorFlow ist ein Framework und Keras die Bibliothek, die eine
Schnittstelle bereitstellt.

In der Dokumentation von TensorFlow gibt es bereits einen beschriebenen
Anwendungsfall, bei dem versucht wird Musik zu erzeugen. Dafür wird ein RNN
aufgebaut, mit dem ein Modell trainiert auf Basis von Piano MIDI Files.

In einem zweiten Anwendungsfall wir mit der LSTM Architektur und einem RNN ein
Modell trainiert, welches Musik generieren kann. Für das Training wird ein
Datensatz von tausenden Irischen Volkslieder verwendet, das in der ABC Notation
vorliegt.

Pydub wurde im Verlauf der Entwicklung für das Zerstückeln ausgewählt, damit ein
stochastischer Ansatz implementiert werden konnte. Hierbei wird anhand von
leisen stellen (definierbar) ein Teilen der Audio Datei vorgenommen. Pydub
ermöglicht weitere Audio manipulationen in Form von Faden, Lautsärkeänderung
etc.

Für das aufteilen in gleich große Stücke wurde die SciPy Bibliothek verwendet.
Der User kann Anhand einer Eingabe die fixe größe der Samples auswählen. SciPy
basiert auf NumPy und wandelt die Audio Daten in ein zahlenbasiertes Format.

-   Die Verwaltung und Dokumentation der Arbeitsschritte finden über GitHub
    statt.

-   Für zusätzliche textuelle Beschreibungen werden Mark-Down-Files verwendet.

-   Die Software Struktur wird mit Hilfe von Unified Modeling Language (UML)
    dargestellt. Hierfür wurden Klassendiagramme, Use Case Diagramme sowie ein
    Komponentendiagramm erstellt. Diese wurden mit der Software StarUML
    erstellt. Die Modellierung der Diagramme stützt sich auf die Literatur von
    Oestereich et al. (Analyse und Design mir der UML 2.5).


**Tensoren** :

Tensoren sind die Datenstruktur, die von maschinellen Lernsystemen verwendet wird. Sie wird als eine Art Container von numerischen Daten angesehen. Ein Tensor hat drei typische **Eigenschaften** : range (Bereich), shape (Form), dtype (Datentyp).

Die Form eines Tensors bezieht sich auf die Anzahl der Dimensionen entlang jeder Achse. Der Datentyp des Tensors ist durch die enthaltenen Datentypen definiert Bsp.: (float32, float64, uint8, int32, int64)

![](Pictures/Tensoren.png)

Aufbau eines Tensors:

tensor.shape(3,4) -> enthält 3 Zeilen mit je 4 Werten

([[2, 10, 20, 22],<br>
 [8, 16, 32, 64],<br>
 [5, 10, 15, 20]])

Algorithmen für maschinelles Lernen verarbeiten jeweils eine Teilmenge von Daten (Charge). Bei der Verwendung eines Datensatzes ist die erste Achse des Tensors für die Größe des Loses (Anzahl der Stichproben) reserviert.

Quellen:

[Introduction to Tensors | TensorFlow Core](https://www.tensorflow.org/guide/tensor)

[Easy TensorFlow - 2- Tensor Types (easy-tensorflow.com)](https://www.easy-tensorflow.com/tf-tutorials/basics/tensor-types)

[TensorFlow Basics: Tensor, Shape, Type, Sessions &amp; Operators (guru99.com)](https://www.guru99.com/tensor-tensorflow.html#:~:text=A%20tensor%20is%20a%20vector,the%20result%20of%20a%20computation.)

**Model:**

**Normalization:**

Quellen:

[Different Types of Normalization in Tensorflow | by Vardan Agarwal | Towards Data Science](https://towardsdatascience.com/different-types-of-normalization-in-tensorflow-dac60396efb0)

**Input Datentypen Tensorflow:**



### Layer in einem künstlichen neuronalen Netzwerk (KNN):

Verschiedene Layer führen unterschiedliche Transformationen an ihren Eingaben durch und einige Layer sind für bestimmte Aufgaben besser geeignet als andere. 
So wird beispielsweise ein convolutional in der Regel in Modellen verwendet, die mit Bilddaten arbeiten. Recurrent Layer werden in Modellen verwendet, die mit Zeitreihendaten arbeiten und fully connected Layer verbinden jede Eingabe vollständig mit jeder Ausgabe innerhalb ihres Layers.

Beispiel für ein Artificial Neural Network: 

![](Pictures/ANN.png)

Hier sieht man, dass der erste Layer, der Input Layer aus drei Knoten (auch Neuronen genannt) besteht. Daraus ergibt sich, dass das Sample aus unserem Datensatz aus drei Dimensionen besteht. Jeder der drei Neuronen in diesem Layer repräsentiert ein einzelnes Feature aus einem bestimmten Sample unseres Datensatzes.  
Jeder der drei Input-Neuronen ist mit jedem Neuron im nächsten Layer verbunden. Jede Verbindung zwischen dem ersten Layer und dem zweiten Layer überträgt die Ausgabe des vorherigen Neurons an den Eingang des empfangenden Neurons (von links nach rechts). Die beiden Layer in der Mitte mit jeweils vier Neuronen sind Hidden Layer, weil sie zwischen dem Input und Output Layer angeordnet sind. 
Die Information wird also durch die Input-Neuronen aufgenommen und durch die Output-Neuronen ausgegeben. Die Hidden-Neuronen liegen dazwischen und bilden innere Informationsmuster ab. Die Neuronen sind miteinander über sogenannte Kanten verbunden. Je stärker die Verbindung ist, desto größer die Einflussnahme auf das andere Neuron. 

Input Layer: Der Input layer versorgt das neuronale Netz mit den notwendigen Informationen. Die Input-Neuronen verarbeiten die eingegebenen Daten und führen diese gewichtet an die nächste Schicht weiter.   

Hidden Layer: Der Hidden Layer befindet sich wie bereits erwähnt zwischen dem Input Layer und dem Output Layer. Während der Input Layer und Output Layer lediglich aus einer Ebene bestehen, können beliebig viele Ebenen an Neuronen im Hidden Layer vorhanden sein. Hier werden die empfangenen Informationen erneut gewichtet und von Neuron zu Neuron bis zum Output Layer weitergereicht. Die Gewichtung findet in jeder Ebene des Hidden Layers statt. Die genaue Prozessierung der Informationen ist jedoch nicht sichtbar. Daher stammt auch der Name, Hidden Layer. Während im Input und Output Layer die eingehenden und ausgehenden Daten sichtbar sind, ist der innere Bereich des Neuronalen Netzes im Prinzip eine Black Box. 

Output Layer: Der Output Layer ist der letzte Layer und schließt unmittelbar an die letzte Ebene des verborgenen Layer an. Die Output-Neuronen beinhalten die resultierende Entscheidung, die als Informationsfluss hervorgeht. 

Wie funktioniert ein KNN? 

Deep Learning ist eine Hauptfunktion eines KNN und funktioniert wie folgt: Bei einer vorhandenen Netzstruktur bekommt jedes Neuron ein zufälliges Anfangsgewicht zugeteilt. Dann werden die Eingangsdaten in das Netz gegeben und von jedem Neuron mit seinem individuellen Gewicht gewichtet. 
Das Ergebnis dieser Berechnung wird an die nächsten Neuronen des nächsten Layer weitergegeben, man spricht auch von einer „Aktivierung der Neuronen“. Eine Berechnung des Gesamtergebnisses geschieht am Output Layer. 
Natürlich sind, wie bei jeder maschinellen Lern-Methode, nicht alle Ergebnisse (Outputs) richtig und es kommt zu Fehlern. Diese Fehler sind berechenbar, ebenso wie der Anteil, den ein einzelnes Neuron an dem Fehler hatte. So wird im nächsten Lern-Durchlauf das Gewicht jedes Neurons so verändert, dass man den Fehler minimiert. 
Anschließend kommt der nächste Durchlauf, indem eine neue Messung des Fehlers nebst Anpassung geschieht. Auf diese Weise „lernt“ das neuronale Netz mit jedem Mal besser, von den Inputdaten auf bekannte Outputdaten zu schließen. 

 

Convolutional Neural Networks (CNN) 

Convolutional Neural Networks (CNN), sind Künstliche Neuronale Netzwerke, die besonders effizient mit 2D- oder 3D-Eingabedaten arbeiten können. Für die Objektdetektion in Bildern verwendet man insbesondere CNNs. 
Der große Unterschied zu den klassischen Neuronalen Netzen besteht in der Architektur von CNNs und damit kann auch der Name „Convolution“ oder „Faltung“ erklärt werden. Der Hidden Layer basiert bei CNNs auf einer Abfolge von Convolution- und Poolingoperationen. Bei der Convolution wird ein so genanntes Kernel über die Daten geschoben und währenddessen eine Faltung gerechnet, was vergleichbar mit einer Multiplikation ist. Die Neuronen werden aktualisiert. Die anschließende Einführung eines Poolinglayers sorgt dafür, dass die Ergebnisse vereinfacht werden. Nur die wichtigen Informationen bleiben anschließend erhalten. Dies sorgt zudem dafür, dass die 2D- oder 3D-Eingangsdaten kleiner werden. Wird dies immer weiter fortgeführt, ergibt sich am Ende in der Ausgabeschicht ein Vektor, der „fully connected layer“. Dieser hat vor allem in der Klassifikation eine besondere Bedeutung, da er so viele Neuronen wie Klassen enthält und die entsprechende Zuordnung über eine Wahrscheinlichkeit bewertet. 

 

Recurrent Neural Networks (RNN) 
Recurrent Neural Networks (RNN) fügen den KNN wiederkehrende Zellen hinzu, wodurch neuronale Netze ein Gedächtnis erhalten. Es ist eine Art von künstlichem neuronalem Netz, das sequentielle Daten oder Zeitreihendaten verwendet. Diese Art von Neuronal Networks wird insbesondere dann verwendet, wenn der Kontext wichtig ist. Denn dann beeinflussen Entscheidungen aus vergangenen Iterationen oder Proben maßgeblich die aktuellen. Mithilfe der Verschaltungen ist es möglich, aus einer Menge von Ausgangsdaten zeitlich codierte Informationen zu gewinnen. Da Recurrent Neural Networks jedoch den entscheidenden Nachteil haben, dass sie über die Zeit instabil werden, ist es inzwischen gang und gäbe so genannte Long Short-Term Memory Units (LSTMs) zu verwenden. Diese stabilisieren das RNN auch für Abhängigkeiten, die über einen längeren Zeitraum bestehen. 


Quellen:

D. (o. D.). Layers in a Neural Network explained. Deeplizard. Abgerufen am 4. Mai 2022, von https://deeplizard.com/learn/video/FK77zZxaBoI

Ognjanovski, G. (2021, 7. Dezember). Everything you need to know about Neural Networks and Backpropagation — Machine Learning Easy and Fun. Medium. Abgerufen am 4. Mai 2022, von https://towardsdatascience.com/everything-you-need-to-know-about-neural-networks-and-backpropagation-machine-learning-made-easy-e5285bc2be3a

Wuttke, L. (2022, 23. März). Künstliche Neuronale Netzwerke: Definition, Einführung, Arten und Funktion. datasolut GmbH. Abgerufen am 4. Mai 2022, von https://datasolut.com/neuronale-netzwerke-einfuehrung/


Literaturverzeichnis:

AudioAnalysis. (o. D.). <https://cs.marlboro.college>. Abgerufen am 5. April
2022, von
<https://cs.marlboro.college/cours/spring2018/jims_tutorials/computational_science/feb26.attachments/AudioAnalysis.html>

Bullock, J. (2021, 13. Dezember). *Simple Audio Processing in Python With Pydub
\- Better Programming*. Medium. Abgerufen am 3. April 2022, von
<https://betterprogramming.pub/simple-audio-processing-in-python-with-pydub-c3a217dabf11>

Chauhan, N. S. (o. D.). *Audio Data Analysis Using Deep Learning with Python
(Part 1)*. KDnuggets. Abgerufen am 5. April 2022, von
<https://www.kdnuggets.com/2020/02/audio-data-analysis-deep-learning-python-part-1.html>

D. (2018). Deep Learning in Natural Language Processing (1st ed. 2018 Aufl.)
[E-Book]. Springer.

datacamp. (o. D.). Using TensorFlow 2.0 to Compose Music Tutorial. DataCamp
Community. Abgerufen am 2. April 2022, von
<https://www.datacamp.com/community/tutorials/using-tensorflow-to-compose-music>

Frochte, J. (2020). Maschinelles Lernen: Grundlagen und Algorithmen in Python
(3., überarbeitete und erweiterte Aufl.). Carl Hanser Verlag GmbH & Co. KG.

Google Colaboratory - Introduction to Deep Learning. (o. D.).
<Https://Colab.Research.Google.Com>. Abgerufen am 3. April 2022, von
<https://colab.research.google.com/github/aamini/introtodeeplearning/blob/master/lab1/solutions/Part2_Music_Generation_Solution.ipynb>

Kersting, K., Lampert, C. & Rothkopf, C. (2019). *Wie Maschinen lernen:
Künstliche Intelligenz verständlich erklärt* (1. Aufl. 2019 Aufl.) [E-Book].
Springer. Abgerufen am 24. März 2022, von
<https://link.springer.com/chapter/10.1007/978-3-658-26763-6_3>

Kleesiek, J., Kaissis, G., Strack, C., Murray, J. M. & Braren, R. (2019). *Wie
funktioniert maschinelles Lernen?* Springer Medizin Verlag.
<https://doi.org/10.1007/s00117-019-00616-x>

Kofler, M. (2021, März). tensorflow-music-generator Public. GitHub. Abgerufen
am 2. April 2022, von
<https://github.com/burliEnterprises/tensorflow-music-generator>

Oestereich, B., Scheithauer, A. & Bremer, S. (2013). *Analyse und Design mit der
UML 2.5: Objektorientierte Softwareentwicklung* (11., umfassend überarbeitete
und aktualisierte Aufl.). De Gruyter Oldenbourg.

Russell, S. & Norvig, P. (2012). *Künstliche Intelligenz: Ein moderner Ansatz
(Pearson Studium - IT)* (3., aktualisierte Aufl.) [E-Book]. Pearson Studium ein
Imprint von Pearson Deutschland.

TensorFlow. (o. D.-a). Generate music with an RNN \| TensorFlow Core.
Abgerufen am 1. April 2022, von
<https://www.tensorflow.org/tutorials/audio/music_generation>

TensorFlow. (o. D.-b). Pitch Detection with SPICE \| TensorFlow Hub. Abgerufen
am 3. April 2022, von <https://www.tensorflow.org/hub/tutorials/spice>

TensorFlow. (o. D.-c). tfio.audio.decode_mp3 \| TensorFlow I/O. Abgerufen am
1\. April 2022, von
[https://www.tensorflow.org/io/api_docs/python/tfio/audio/decode_mp3\#args](https://www.tensorflow.org/io/api_docs/python/tfio/audio/decode_mp3#args)

Tutorialink. (o. D.). Split audio files using silence detection – Python.
Tutorialink.Com. Abgerufen am 6. April 2022, von
<https://python.tutorialink.com/split-audio-files-using-silence-detection/>



