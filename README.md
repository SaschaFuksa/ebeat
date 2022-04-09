# <h1>ebeat project</h1>
Projektmanagement: [Trello Board](https://trello.com/b/HYG7tuaq/tech-lab) 

<p>Patrick Treiber 42609</p>
<p>Michael Henn </p>
<p>Vanessa Hartmann 42580 </p>
<p>Nicola Haller 42617 </p>
<p>Sascha Fuska 42590 </p>
<p>Arline Carle 42582 </p>

## <h2> Anforderunsganalyse </h2>
### <h3> Ziel der Anforderung: </h3>
Das Ziel der zu entwickelnden Maschine ist, dass sie automatisiert Musik zusammenstellt.

### <h3> Anwendungsbereich:  </h3>

Der Anwendungsbereich einer zu entwickelnden KI ist, dass die KI selbstständig (automatisiert) Musikstücke zerlegt und wieder zu neuen Musikstücken zusammenstellt.  


### <h3>Use Case   </h3> 

 <U><h4> ebeat – Samplecreator </h4></U>

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
#### <h4><U> ebeat – ML Trainer </h4></U>
 
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


#### <h4><U> ebeat-Resampler </h4></U>

## <h2> Systemarchitektur</h2>
### <h3>Komponentendiagramm  </h3>
### <h3>Klassendiagramm  </h3>
#### <h4><U> Audio Component </h4></U>
#### <h4><U> ML-Component </h4></U>
#### <h4><U> Complier Component </h4></U>

## <h2>Theoretische Grundlagen</h2>

## <h2> Systemspezifikation</h2>

