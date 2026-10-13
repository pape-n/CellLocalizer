"""Short guide in the program, English and German (CLAUDE.md, phase 5)."""

from __future__ import annotations

from qtpy.QtWidgets import QDialog, QHBoxLayout, QPushButton, QTextBrowser, QVBoxLayout

GUIDE_EN = """
<h2>CellLocalizer: how to use it</h2>
<p>The program measures where a protein (the <b>target</b>, e.g. titin) sits in a cell: in the nucleus,
in the cytoplasm, in the actin part of the cytoplasm, near the nuclear envelope, under the cell surface
and near the coverslip. It works on 3D stacks (Zeiss CZI, Nikon ND2 or TIFF) and always measures the raw values.
How every value is computed, with the literature, is in Help &gt; <b>Methods text</b>; that text is also the
methods section for a thesis or paper.</p>

<h3>Start page and navigation</h3>
<p>The start page offers <b>New analysis</b>, <b>Open analysis</b> (with the last analyses), the guide, the
methods text and <b>Settings</b> (the defaults every new analysis starts with, and the text size).
<b>Home</b> brings you back; the analysis stays open. In the image window the bar with the five steps can be
clicked: finished steps open at once, later steps when everything before them is done (a locked step says
why). Keyboard: Ctrl+Right and Ctrl+Left for Next and Back, in step 5 Ctrl+Enter accepts, Ctrl+Shift+Right
opens the next image.</p>

<h3>1 Open</h3>
<p><b>Add folder</b> adds every image of an experiment day, also from subfolders (Cell1, Cell2, ...).
Raw Airyscan data, time series, multi-position files and TIFF exports next to CZI or ND2 files are skipped. Experiment and condition are
filled in from the folder and file names; check them. <b>Exclude</b> leaves an image out, with a reason.
After opening, the program stays in step 1 so you can look at the image; then click <b>Next</b>.</p>
<p><b>Blind review</b> (on by default, Settings): images get a random code such as "Image 07", and file
name, folder and condition stay hidden while you check cells, so the condition cannot steer the decision.
<b>Show name</b> reveals one image and is noted in the analysis file. Results.xlsx shows all names.</p>

<h3>2 Channels</h3>
<p>Tell the program what each channel shows: Nucleus (DAPI), Actin (phalloidin), Target (the protein
you want to localize, several are possible), Comparison (a second stained protein, e.g. Vimentin: it is
measured like a target, and every target is measured in its stained area as well), Membrane (a membrane stain: then only it defines the cell
edge), Other (shown and used for the cell edge, not measured), Ignore. The program guesses the roles from
the file name (e.g. "Act488 TTN647"); check them. <b>Save as panel</b> remembers them for all images with
the same channels.</p>

<h3>3 Cell</h3>
<p>The white line is the cell edge. Each stain counts up to the point where it fades into the dark
background (Li threshold per channel), so the edge lies outside the actin cortex. Settings can switch to
"Half maximum", which trims the blur but can cut through a dim cortex in the upper planes. If a neighbour cell or debris is attached, press <b>Draw</b>, click
around it, double-click, then <b>Apply</b>: it turns black in all planes. The <b>edge factor</b> makes the
cell larger (below 1) or smaller (above 1). Faint signal above thin protrusions is mostly blur from the
planes below; lowering the factor counts that blur as cell. Where exactly the edge lies changes the
enrichment values (the cell volume is in the denominator); the <b>sensitivity span</b> in step 5 shows by how
much.</p>

<h3>4 Compartments</h3>
<p>Show the masks one by one. Untick the image channels to see the masks alone. In 3D the masks are drawn
as smoothed surfaces. Adjust the edges with the factors. <b>Use these factors for all cells</b> makes
them the starting values of the analysis. Actin is everything above the threshold of the whole cell or above
the threshold of its own plane (halfway between cytoplasm and actin of that plane), so the dimmer cortex in
the upper planes is found as well as the bright layer at the coverslip.</p>
<p>The <b>target mask</b> marks the stained voxels. By default the shares count only the target signal inside
it (Settings > Masks > "Target signal counted"), so correcting the mask with its factor changes the numbers;
the sensitivity span then shows how much. A <b>comparison mask</b> (e.g. Vimentin) is the stained area of a
comparison channel. The orange arrow (<b>Front direction</b>) points from the cell centre to the actin-rich part;
it is long and bold for a clear front and short and thin when the cell is nearly even.</p>

<h3>5 Results</h3>
<p><b>Signal %</b>: share of the target's signal in a region. <b>Volume %</b>: share of the cell volume.
<b>Enrichment</b> = Signal % / Volume %: 1 means as dense as the cell average, above 1 enriched. Nucleus
and cytoplasm add up to 100 %. Actin is the actin part of the cytoplasm. An empty field means no value,
e.g. without a nucleus channel. Accept the cell, or <b>Exclude</b> it with a reason, then <b>Next image</b>.</p>
<p><b>Zones</b> overlap the compartments: nuclear envelope zone (both sides of the nucleus edge, also split
into the inner and the outer half), nucleus interior, cortex shell under the cell surface, basal layer above
the coverslip. The <b>highest enrichment at ... µm from the nucleus edge</b> says on which side of the edge the
signal peaks, averaged over the whole nucleus (negative = inside). A single spot closer to the edge than the
resolution (about 0.15 µm in the plane, 0.4 µm in z) cannot be placed inside or outside; a Lamin B1 stain as
comparison channel is the best reference. <b>Front half</b>: the half of the cell towards the actin front;
the <b>polarity index</b> (about 0 = no front) says how clear that front is, below the threshold in the
settings the front values mean little. <b>Rear to front</b>: density in the rear half divided by the density
in the front half (above 1 = more in the rear). <b>PCC</b> (Pearson, -1 to 1) and <b>M1/M2</b> (Manders) describe the colocalization with actin.
The orange <b>notes</b> point to things to check, e.g. a second nucleus in the cell, DAPI outside the
nucleus mask, a target without staining, a very high correlation with actin (crosstalk?), microscope
or settings that differ within an experiment. Results.xlsx also gives <b>bottom_plane_ratio</b>: near 1 the
lowest plane is already the brightest, the stack starts inside the cell and the basal layer is only the lowest
recorded layer. Start new stacks about 1 µm below the cell. The <b>sensitivity span</b> gives the enrichment and signal share
in the nucleus when the cell and nucleus edges move (threshold x0.8 to x1.2): a wide span means the value
depends on the edge, look at the cell again.</p>

<h3>Channel offset (once per microscope)</h3>
<p>Different colours are focused at slightly different heights. Heights and thin zones (basal layer, cortex)
depend on this offset. Record multicolour beads (e.g. 0.1 µm TetraSpeck) with the same settings as the cells,
then Settings &gt; <b>Channel offset</b> &gt; <b>Measure from a bead image</b>. The channels are then moved
back by whole voxels before every measurement, also on screen. Measured on the start page (Settings there),
the offset is stored per fluorophore and used in every new analysis.</p>

<h3>Many images</h3>
<p><b>Analyze all images</b> (step 1) computes every image of the list that has no cell yet, with the
current settings. It needs the channel roles: open one image and <b>save a panel</b> in step 2 first.
The cells are marked <i>to check</i>; <b>Next image</b> opens them one after the other. Only checked
(accepted) cells go into the Prism sheets. After a change in <b>Settings</b>, <b>Apply settings to all
cells</b> (step 5) computes every cell again whose settings differ.</p>

<h3>Saving</h3>
<p>The first <b>Save results</b> (step 5, bottom right) asks for a folder. It then holds <b>Results.xlsx</b> (start with the
sheet <i>Read me</i>; one sheet per number in Prism layout), one control image per cell and
<b>analysis.lta.json</b> to open the analysis again. After that every cell is saved automatically.</p>

<h3>Cell viewer: pictures for figures</h3>
<p><b>Cell viewer</b> on the start page (or File > View and export cells once the analysis is saved) opens a
saved analysis.
Choose a cell on the left. <b>Plane</b> shows one recorded plane, <b>Section</b> a vertical cut through the
cell at any direction (coverslip at the bottom, stretched to the true height), <b>3D</b> the whole cell (drag to
turn). Switch channels, colours, contrast, masks and zones on and off. <b>Save picture</b> writes a PNG with
scale bar (and legend or title if wanted) in black margins, <b>Save values</b> a TIFF with the recorded values of
the shown plane or section for your own processing. <b>Export all cells</b> saves the same view for every cell,
one folder per condition, with one contrast for all cells so that brightness stays comparable, plus the
settings used. The masks are computed once with the settings of the analysis and kept in the folder
"Cell masks". Nothing in the viewer changes a measured value.</p>

<h3>Practice</h3>
<p><b>Practice with a synthetic cell</b> (start page, bottom, or Help) opens a computed example cell to try
every step.</p>

<h3>If something goes wrong</h3>
<p>Help > <b>Report a problem</b> says what to report, where to send it, and copies the details of your computer.
After a crash the program shows where the crash log is. Please send it along.
Results.xlsx cannot be written while it is open in Excel; close it, the next save writes it.</p>
"""

GUIDE_DE = """
<h2>CellLocalizer: Bedienung</h2>
<p>Das Programm misst, wo ein Protein (das <b>Target</b>, z. B. Titin) in einer Zelle liegt: im Kern, im
Zytoplasma, im Aktinanteil des Zytoplasmas, an der Kernhülle, unter der Zelloberfläche und am Deckglas.
Es arbeitet mit 3D-Stapeln (Zeiss CZI, Nikon ND2 oder TIFF) und misst immer die Rohwerte. Wie jeder Wert berechnet wird,
mit Literatur, steht unter Help &gt; <b>Methods text</b>; dieser Text ist zugleich der Methodenteil für Doktorarbeit
oder Paper.</p>

<h3>Startseite und Navigation</h3>
<p>Die Startseite bietet <b>New analysis</b>, <b>Open analysis</b> (mit den letzten Analysen), die Anleitung,
den Methodentext und <b>Settings</b> (die Standardwerte für neue Analysen und die Schriftgröße). <b>Home</b>
führt zurück, die Analyse bleibt offen. Im Bildfenster lässt sich die Leiste mit den fünf Schritten anklicken:
erledigte Schritte öffnen sofort, spätere, sobald alles davor erledigt ist (ein gesperrter Schritt sagt warum).
Tastatur: Strg+Rechts und Strg+Links für Weiter und Zurück, in Schritt 5 nimmt Strg+Enter an,
Strg+Umschalt+Rechts öffnet das nächste Bild.</p>

<h3>1 Öffnen</h3>
<p><b>Add folder</b> nimmt alle Bilder eines Versuchstags auf, auch aus Unterordnern (Cell1, Cell2, ...).
Airyscan-Rohdaten, Zeitserien, Dateien mit mehreren Positionen und TIFF-Exporte neben CZI- oder ND2-Dateien werden übersprungen. Experiment und Bedingung
werden aus Ordner- und Dateinamen vorbelegt, bitte prüfen. <b>Exclude</b> schließt ein Bild mit Grund aus.
Nach dem Öffnen bleibt das Programm in Schritt 1, damit man das Bild ansehen kann, dann <b>Next</b>.</p>
<p><b>Blindes Prüfen</b> (Standard an, Settings): Bilder bekommen eine zufällige Kennung wie „Image 07“,
Dateiname, Ordner und Bedingung bleiben beim Prüfen verborgen, damit die Bedingung die Entscheidung nicht
lenkt. <b>Show name</b> zeigt ein Bild mit Namen und wird in der Analysedatei vermerkt. Results.xlsx zeigt
alle Namen.</p>

<h3>2 Kanäle</h3>
<p>Für jeden Kanal angeben, was er zeigt: Nucleus (DAPI), Actin (Phalloidin), Target (das gesuchte
Protein, mehrere möglich), Comparison (ein zweites gefärbtes Protein, z. B. Vimentin: es wird wie ein Target
gemessen, und jedes Target wird zusätzlich in seiner gefärbten Fläche gemessen), Membrane (Membranfärbung, dann bestimmt nur sie den Zellrand), Other (wird
gezeigt und für den Zellrand genutzt, nicht gemessen), Ignore. Das Programm rät die Rollen aus dem
Dateinamen (z. B. "Act488 TTN647"), bitte prüfen. <b>Save as panel</b> merkt sie sich für alle Bilder mit
denselben Kanälen.</p>

<h3>3 Zelle</h3>
<p>Die weiße Linie ist der Zellrand. Jede Färbung zählt bis dorthin, wo sie im dunklen Hintergrund
verschwindet (Li-Schwelle pro Kanal), der Rand liegt also außen am Aktinkortex. In den Einstellungen lässt sich
„Half maximum“ wählen, das die Unschärfe abschneidet, aber in oberen Ebenen durch einen schwachen Kortex
schneiden kann. Hängt eine Nachbarzelle oder Schmutz daran: <b>Draw</b>, drumherum
klicken, Doppelklick, dann <b>Apply</b>. Der Bereich wird in allen Ebenen schwarz. Der <b>Edge factor</b>
macht die Zelle größer (unter 1) oder kleiner (über 1). Schwaches Signal über dünnen Ausläufern ist
meist Unschärfe aus den Ebenen darunter, ein kleinerer Faktor zählt diese Unschärfe zur Zelle. Wo genau der
Rand liegt, verändert die Anreicherungen (das Zellvolumen steht im Nenner), die <b>Empfindlichkeitsspanne</b> in
Schritt 5 zeigt, wie stark.</p>

<h3>4 Kompartimente</h3>
<p>Masken einzeln ein- und ausblenden. Die Bildkanäle lassen sich abschalten, dann sieht man nur die
Masken. In 3D werden die Masken als geglättete Flächen gezeigt. Ränder über die Faktoren anpassen.
<b>Use these factors for all cells</b> macht sie zu Startwerten der Analyse. Aktin ist alles über der Schwelle
der ganzen Zelle oder über der Schwelle der eigenen Ebene (auf halbem Weg zwischen Zytoplasma und Aktin dieser
Ebene). So wird der schwächere Kortex in den oberen Ebenen ebenso erfasst wie die helle Schicht am Deckglas.</p>
<p>Die <b>Target-Maske</b> markiert die gefärbten Voxel. Standardmäßig zählen die Anteile nur das Targetsignal in
ihr (Settings > Masks > "Target signal counted"), eine Korrektur der Maske über ihren Faktor ändert also die
Zahlen, die Empfindlichkeitsspanne zeigt, wie stark. Eine <b>Comparison-Maske</b> (z. B. Vimentin) ist die
gefärbte Fläche eines Vergleichskanals. Der orange Pfeil (<b>Front direction</b>) zeigt vom Zellmittelpunkt
zum aktinreichen Teil, lang und kräftig bei klarer Front, kurz und dünn bei einer fast gleichmäßigen Zelle.</p>

<h3>5 Ergebnisse</h3>
<p><b>Signal %</b>: Anteil des Targetsignals in einem Bereich. <b>Volume %</b>: Anteil am Zellvolumen.
<b>Enrichment</b> = Signal % / Volume %: 1 heißt so dicht wie im Mittel der Zelle, über 1 angereichert.
Kern und Zytoplasma ergeben zusammen 100 %. Aktin ist der Aktinanteil des Zytoplasmas. Ein leeres Feld
heißt: kein Wert, z. B. ohne Kernkanal. Zelle annehmen oder mit Grund ausschließen (<b>Exclude</b>), dann
<b>Next image</b>.</p>
<p><b>Zonen</b> überlappen die Kompartimente: Kernhüllenzone (beidseits des Kernrands, zusätzlich in innere
und äußere Hälfte geteilt), Kerninneres, Kortexschale unter der Zelloberfläche, Basalschicht über dem Deckglas.
Die <b>höchste Anreicherung bei ... µm vom Kernrand</b> sagt, auf welcher Seite des Rands das Signal am
stärksten ist, gemittelt über den ganzen Kern (negativ = innen). Ein einzelner Punkt näher am Rand als die
Auflösung (etwa 0,15 µm in der Ebene, 0,4 µm in z) lässt sich nicht innen oder außen zuordnen, die beste
Referenz ist eine Lamin-B1-Färbung als Vergleichskanal. <b>Front half</b>: die Zellhälfte zur Aktinfront hin,
der <b>Polaritätsindex</b> (etwa 0 = keine Front) sagt, wie deutlich diese Front ist, unter der Schwelle in den
Einstellungen sagen die Frontwerte wenig. <b>Rear to front</b>: Dichte in der hinteren Hälfte geteilt durch die
Dichte in der vorderen (über 1 = mehr hinten).
<b>PCC</b> (Pearson, -1 bis 1) und <b>M1/M2</b> (Manders) beschreiben die Kolokalisation mit Aktin. Die
orangen <b>Hinweise</b> nennen, was zu prüfen ist, z. B. ein zweiter Kern in der Zelle, DAPI außerhalb der
Kernmaske, ein Target ohne Färbung, eine sehr hohe Korrelation mit Aktin (Übersprechen?),
oder Mikroskopeinstellungen, die sich innerhalb eines Experiments unterscheiden. Results.xlsx enthält außerdem
<b>bottom_plane_ratio</b>: nahe 1 ist schon die unterste Ebene die hellste, der Stapel beginnt in der Zelle, und die
Basalschicht ist nur die unterste aufgenommene Schicht. Neue Stapel etwa 1 µm unterhalb der Zelle beginnen. Die <b>Empfindlichkeitsspanne</b>
gibt Anreicherung und Signalanteil im Kern an, wenn Zell- und Kernrand wandern (Schwelle x0,8 bis x1,2): eine
breite Spanne heißt, der Wert hängt am Rand, die Zelle noch einmal ansehen.</p>

<h3>Kanalversatz (einmal pro Mikroskop)</h3>
<p>Verschiedene Farben werden in leicht verschiedenen Höhen scharf abgebildet. Höhen und dünne Zonen
(Basalschicht, Kortex) hängen von diesem Versatz ab. Mehrfarben-Beads (z. B. 0,1 µm TetraSpeck) mit denselben
Einstellungen wie die Zellen aufnehmen, dann Settings &gt; <b>Channel offset</b> &gt; <b>Measure from a bead
image</b>. Die Kanäle werden dann vor jeder Messung um ganze Voxel zurückgeschoben, auch in der Ansicht. Auf
der Startseite gemessen (Settings dort), gilt der Versatz pro Farbstoff für jede neue Analyse.</p>

<h3>Viele Bilder</h3>
<p><b>Analyze all images</b> (Schritt 1) rechnet alle Bilder der Liste, die noch keine Zelle haben, mit
den aktuellen Einstellungen. Dafür braucht es die Kanalrollen: vorher ein Bild öffnen und in Schritt 2
ein <b>Panel speichern</b>. Die Zellen sind dann <i>to check</i>, <b>Next image</b> öffnet sie
nacheinander. Nur geprüfte (angenommene) Zellen kommen in die Prism-Blätter. Nach einer Änderung unter
<b>Settings</b> rechnet <b>Apply settings to all cells</b> (Schritt 5) alle Zellen mit abweichenden
Einstellungen neu.</p>

<h3>Speichern</h3>
<p>Beim ersten <b>Save results</b> (Schritt 5, unten rechts) einen Ordner wählen. Darin liegen <b>Results.xlsx</b> (mit dem Blatt
<i>Read me</i> beginnen, je Kennzahl ein Blatt im Prism-Format), ein Kontrollbild pro Zelle und
<b>analysis.lta.json</b> zum Wiederöffnen. Danach wird jede Zelle automatisch gespeichert.</p>

<h3>Zellbetrachter: Bilder für Abbildungen</h3>
<p><b>Cell viewer</b> auf der Startseite (oder File > View and export cells, sobald die Analyse gespeichert ist)
öffnet eine gespeicherte Analyse. Links eine Zelle wählen. <b>Plane</b> zeigt eine aufgenommene Ebene, <b>Section</b> einen
senkrechten Schnitt durch die Zelle in beliebiger Richtung (Deckglas unten, auf die wahre Höhe gestreckt),
<b>3D</b> die ganze Zelle (zum Drehen ziehen). Kanäle, Farben, Kontrast, Masken und Zonen lassen sich ein- und
ausschalten. <b>Save picture</b> speichert ein PNG mit Maßstab (und auf Wunsch Legende oder Titel) in schwarzen
Rändern, <b>Save values</b> ein TIFF mit den aufgenommenen Werten der gezeigten Ebene oder des Schnitts zur
eigenen Weiterverarbeitung. <b>Export all cells</b> speichert dieselbe Ansicht für jede Zelle, ein Ordner pro
Bedingung, mit einem Kontrast für alle Zellen, damit die Helligkeit vergleichbar bleibt, dazu die verwendeten
Einstellungen. Die Masken werden einmal mit den Einstellungen der Analyse berechnet und im Ordner
„Cell masks“ abgelegt. Der Betrachter verändert keinen Messwert.</p>

<h3>Üben</h3>
<p><b>Practice with a synthetic cell</b> (Startseite unten oder Help) öffnet eine berechnete Beispielzelle, mit der
sich jeder Schritt ausprobieren lässt.</p>

<h3>Wenn etwas schiefgeht</h3>
<p>Help > <b>Report a problem</b> sagt, was in eine Rückmeldung gehört und wohin sie geht, und kopiert die Angaben zum Rechner.
Nach einem Absturz zeigt das Programm, wo das Absturzprotokoll liegt. Bitte mitschicken.
Results.xlsx lässt sich nicht schreiben, solange die Datei in Excel offen ist. Excel schließen, beim
nächsten Speichern wird sie geschrieben.</p>
"""


class HelpDialog(QDialog):
    def __init__(self, parent=None, language: str = "en"):
        super().__init__(parent)
        from localization_tool.gui import theme

        theme.style(self)
        self.setWindowTitle("How to use / Bedienung")
        theme.fit_to_screen(self, 720, 640)
        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        self.en = QPushButton("English")
        self.de = QPushButton("Deutsch")
        for b in (self.en, self.de):
            b.setCheckable(True)
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        self.text = QTextBrowser()
        lay.addWidget(self.text, 1)
        self.en.clicked.connect(lambda: self.show_language("en"))
        self.de.clicked.connect(lambda: self.show_language("de"))
        self.show_language(language)

    def show_language(self, language: str) -> None:
        self.text.setHtml(GUIDE_DE if language == "de" else GUIDE_EN)
        self.en.setChecked(language != "de")
        self.de.setChecked(language == "de")
