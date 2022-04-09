# Anfoderungsanalyse

##Anforderungen

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

##Systemspezifikation

**1. Audiokomponente:**

-   Laden der Audio-Files

-   Audio-Files müssen sinnvoll geteilt werden (Samples), manuell oder
    stochastisch

-   Nummerierung/Benennung für Zuordnung der Audio-Files notwendig

-   Samples müssen zu einem Stream/Audio-File zusammengesetzt werden können

-   Speicherung des Audio Streams/Audio-Files

-   Kann Audio-Files (WAV/MP3 -\> Format egal) verwenden

-   Audioeigenschaften können analysiert werden -\> Metadaten

-   Erzeugung eines Streams/Audio-Files (Reihenfolge) anhand einzelner Samples

-   Speichern der Audiofiles

**2. Compilerkomponente:**

-   Samples werden in mathematischer Form = Zahlenformat konvertiert
    (Tabelle/Matrix)

-   Übersetzt Samples anhand eines geeigneten Formates -\> Tensor für
    ML-Komponente

-   Leitet die übersetzten Formate an ML-Komponente weiter

**3. Inferenzkomponente / ML-Komponente (Machine Learning):**

-   Lernalogrythmus identifizieren und bewertet ob einzelne Samples harmonieren
    \-\> Notwendig für das Erstellen des Streams/Audio-Files, dies muss von
    Audio-Komponente rückgemeldet werden um die einzelnen Samples
    zusammenzuführen

-   RNN / LSTM wird anhand Tensorflow Machine Learning Plattform umgesetzt

-   Audioeigenschaften (Amplitude -\> Lautstärke, Wellenlänge -\> Schnelligkeit,
    Phase -\> Verschiebung) werden sinnvoll betrachtet und ggf. verwendet

-   *Speicherung der neuen Musikstücke*
