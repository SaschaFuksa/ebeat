# ebeat project

## Anforderunsganalyse
###Ziel der Anforderung: 
Das Ziel der zu entwickelnden Maschine ist, dass sie automatisiert Musik zusammenstellt. 

###Anwendungsbereich: 

Der Anwendungsbereich einer zu entwickelnden KI ist, dass die KI selbstständig (automatisiert) Musikstücke zerlegt und wieder zu neuen Musikstücken zusammenstellt.  

###Use Case  

####ebeat – Samplecreator 

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

####eBeat – ML Trainer 

## Theoretische Grundlagen

## Systemspezifikation
