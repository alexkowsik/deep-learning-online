---
title: Residual Neural Networks
subtitle: Trainieren von sehr tiefen Netzen
author: Alexander Kowsik
date: 20.01.2021

header-includes:
    <link rel="shortcut icon" type="image/x-icon" href="images/favicon.ico">
---

<div id="vid">
<iframe src="https://drive.google.com/file/d/1lWTWhwSrc8Gqd-9N4KbeXVkC3J0RkWnk/preview" height="100%" width="100%"></iframe></div>


## Motivation

Bei <span class="mark">**Residual Neural Networks**</span>, oder kurz _ResNets_, handelt es sich um eine bestimmte Art von neuronalen Netzen, die sich dadurch auszeichnen, dass sie sehr viele hidden layer besitzen, also äußerst 'tief' sind. Dabei sind sie trotzdem noch sehr effizient trainierbar und erzielen hohe Performances. Besondere Anwendung finden sie im Bereich der Bildklassifikation in Verbindung mit Convolutional Neural Networks (CNNs).  
Bevor wir uns genauer anschauen, wie ResNets aussehen und funktionieren, betrachten wir zunächst das zentrale Problem, welches die Entwicklung von ResNets motiviert hat: das **Trainieren von sehr tiefen Netzen**. "Sehr tief" ist dabei im folgenden alles über ca. 25 hidden layer. Wir schauen uns an, warum das in der Praxis normalerweise nur begrenzt möglich ist und wie ResNets das Problem lösen können.

### Warum möchte man _tiefe_ neuronale Netze?

Um zu verstehen, warum tiefe Netze oftmals bessere Performances liefern als weniger tiefe Netze, ist es hilfreich, sich noch einmal vor Augen zu führen, wie und warum ein tiefes neuronal Netz überhaupt funktioniert, hier mal am Beispiel von **Convolutional Neural Networks**.

![Die layer eines CNNs visualisiert](images/deepNetVis.png){ width=90% }

Ein ConvNet besteht in der Regel aus mehrern hintereinander geschalteten convolutional layern mit anschließenden pooling- und batch-normalization layern. Am Ende des Netzwerkes finden sich meistens einige wenige fully-connected layer, die beispielsweise in einem Ausgabe-layer zur Klassifikation enden können. Die Idee für ConvNets entspringt aus den Erkenntnissen der klassischen Computergrafik, da die zu lernende Gewichte eines ConvNets die Parameter von linearen Filtern sind, die genutzt werden können, um Strukturen in einem Bild zu erkennen.

Dabei ist es so, dass die gelernten features von Schicht zu Schicht immer abstrakter werden. Man kann es sich so vorstellen, dass die ersten layer lernen, Kanten und Ecken zu erkennen, weitere layer lernen, diese Kanten zu Texturen oder Formen zusammenzusetzen und somit Objektteile zu erkennen, und die letzten Schichten lernen, wie sich diese Objektteile zueinander verhalten und zu ganzen Objekten zusammensetzen. Dadurch wird dann Bilderkennung möglich. In der obigen Abbildung sind die einzelnen Schichten eines ConvNets visualisiert und zeigen diesen Umstand noch einmal eindrücklich, auch wenn es nicht immer der Fall sein muss, dass diese features visuell gut von uns Menschen interpretiert werden können.

Theoretisch ist es möglich, mit nur wenigen layern Klassifikation lediglich anhand der ersten low level features wie den Kanten vorzunehmen (_'Universal approximation theorem'_), jedoch ist es praktisch nur sehr schwer umzusetzen, da dies extrem viele Trainingsbeispiele und sehr lange Trainingszeit erfordert. Daher ist das Hinzufügen von weiteren layern nur sinnvoll. Es erlaubt es, auf eine effizientere Weise mehr und abstraktere (somit oft auch bessere) features zu lernen.

Dies ist nicht nur bei Bilderkennung mit ConvNets der Fall. Neuronale Netze sind grundsätzlich **universelle Funktionsapproximatoren**. <span class="mark">Je mehr layer ein neuronales Netz hat, desto einfacher ist es, immer komplexere Funktionen zu lernen und sich immer mehr der zu lernenden Idealfunktion anzunähern.</span>

Dieser Trend, immer mehr layer zu benutzen um die Performance für eine bestimmte Aufgabe zu steigern, hat sich in der Vergangenheit auch gezeigt.

![ImageNet competition: Test error (blau) und Anzahl der layer (orange)](images/imagenet_competition.png){ width=60% }

Betrachtet man die **ImageNet competition** bist zum Jahr 2015, so stellt man fest, dass die Architekturen mit den besten Ergebnissen von Jahr zu Jahr immer tiefer wurden. Auch an anderen Datensätzen war dies erkennbar - tiefere Netze erreichten bessere Genauigkeiten.

Mit diesen Beobachtungen könnte man schlussfolgern, dass tiefe Netze immer besser sind, also die Lösung für schlechte Performance einfach stets "mehr layer" sein könnte. Doch dem ist leider im Allgmeinen nicht so.

### Das Problem mit sehr tiefen Netzen

Es hat sich herausgestellt, dass das Training von sehr tiefen Netzen, also von Netzen mit über ungefähr 25 oder 30 layern, Probleme bereitet und nicht mehr ohne weiteres möglich ist. Warum ist das so?

![Degeneration Problem<sup>1</sup>](images/degeneration.jpg){ width=85% }

<span class="mark">Der Hauptgrund dafür ist ein Umstand der als **'degeneration problem'** bekannt ist.</span> Das Hinzufügen von weiteren layern verbessert zwar zunächst die Trainierbarkeit und Performance eines Netzes, letztere erreicht erwartungsgemäß irgendwann ein gewisses Plateau, fängt jedoch ab einem bestimmten Punkt an, stark abzufallen. Das heißt, insbesondere sind bei sehr tiefen Netzen sowohl der Trainings Error als auch der Test Error viel höher als bei gleichen, weniger tiefen Netzen.

Eine Erklärung dafür könnte **Overfitting** sein. Vielleicht ist das Netz ab einem bestimmten Punkt so komplex, dass es anfängt, sich zu stark an den gegebenen Trainingsdatensatz anzupassen, ihn auswendig zu lernen oder sich zu sehr auf Rauschen in den Daten zu konzentrieren, sodass es nicht mehr gut generalisiert. Overfitting kann man hier jedoch ausschließen, da nicht nur der Test Error, sondern auch der Trainings Error höher ist, das Netz also noch nicht einmal die Trainingsdaten gut erfasst. Das bedeutet auch, dass weder Dropout noch andere Regularisierungsmaßnahmen gegen das degeneration problem helfen.

Eine weitere Erklärung ist das Problem von **vanishing/exploding gradients**. Da im backpropagation Schritt die Gradienten der loss-Funktion mithilfe die Kettenregel ermittelt werden und somit bei tiefen Netzen sehr viele Terme miteinander multipliziert werden, kann das Ergebnis bei sehr kleinen Zwischenwerten sehr klein (oder 0) und bei großen sehr groß werden, sodass die Konvergenz behindert wird und das Training somit fehlschlägt. Dieses Problem ist tatsächlich manchmal der Grund, warum das degeneration problem auftritt, jedoch lässt sich zeigen, dass selbst durch Nutzung von batch normalization und Ausschließen von vanishing/exploding gradients das degeneration problem in der Praxis trotzdem weiterhin auftritt.

![Tiefe Netze: zumindest die gleiche Performance wie weniger tiefe?](images/AB.png){ width=65% }

Es hat stattdessen viel mehr mit dem **Optimierungsmechanismus** zu tun, und damit, wie die **Parameter initialisiert** werden. Theoretisch sollte es nämlich so sein, dass ein tieferes Netz B mit n layern wenigstens genau so gute Ergebnisse erzielt wie ein gleiches, nur weniger tiefes Netz A mit m < n layern. Dies kann man sich auf folgende Weise klarmachen: das tiefere Netz B könnte in seinen ersten layern das gleiche lernen wie A, somit wären die ersten m layer die gleichen wie bei A. Bei allen folgenden layern würde B die Identitätsfunktion lernen und damit die exakt gleichen Ergebnisse liefern wie A.

Doch ganau hier liegt das Problem - <span class="mark">die Identität zu lernen ist in der Regel nur sehr schwierig</span>. Bei der Initialisierung werden die Parameter nämlich in der Regel aus einer Gaußverteilung mit Mittelwert 0 gezogen. Das heißt, für einen 'solver', also Lösealgorithmus, ist es relativ einfach, die Nullfunktion anzunähern, da die Parameter bereits relativ nahe um die 0 verteilt sind. Es ist für ihn jedoch schwer, die Identität zu lernen, genauer gesagt genauso schwer wie jede andere Funktion.  
Ein Beispiel: bei einem 3x3 linearen Filter wäre die Identitätsfunktion

![Identität bei 3x3 convolutions](images/3x3.png){ width=20% }

damit müssen alle 9 Gewichte richtig gelernt werden. Ein solver findet in der gegebenen Trainingszeit und mit den vorhanden Trainingsdaten somit meistens keinen Weg dazu. <span class="mark">Dadurch entstehen durch mehr layer auch solche layer, die der Performance des gesamten Netzes schaden, anstatt sie zu verbessern.</span>

Genau hier setzen die Residual Networks an und bieten eine Lösung für eben dieses Problem.

## Residual Networks

### Das Lernen von Residuen

Die Idee, die ermöglicht, einfacher die Identitätsfunktion zu lernen, ist folgende: anstatt zu hoffen, dass alle paar hintereinander geschaltetete layer eine zugrunde liegende Funktion direkt approximieren, werden bei ResNets in den layern nur die **Residuen** der Eingabe in dieses layer zu der Idealfunktion explizit gelernt, daher auch der Name _Residual Networks_. Was genau bedeutet das?

Noch einmal zur Erinnerung: der Begriff _Residuum_ bezeichnet die Abweichung eines Datenpunktes von dem vom Modell geschätzten Wert. Im Beispiel von linearer Reagression im R<sup>2</sup> wäre dies der vertikale Abstand eines Punktes zur geschätzten Regressionsgerade.

Betrachten wir uns einige hinterinander geschaltete layer eines feed-foward Netzwerkes. Sei _x_ dabei die Eingabe in diese layer und _H(x)_ die zugrunde liegende Funktion, die von diesen layern gefitted werden soll. Anstatt diese layer direkt _H(x)_ approximieren zu lassen, lernen die layer in ResNets die Restfunktion **_F(x) = H(x) - x_**, also lediglich die Abweichung der Idealfunktion von der Eingabe _x_, umgeschrieben ist die Idealfunktion also _H(x) = x + F(x)_. Beide Ansätze approximieren asymptotisch die gewünschte Funktion, <span class="mark">jedoch hat sich herausgestellt, dass das Lernen von Residuen einfacher zu sein scheint und eine Reihe Vorteile mit sich bringt.</span>

Der Hauptgrund dafür wird ersichtlich, wenn wir uns die Motivation für ResNets noch einmal anschauen. Es ist wie in den obigen Kapiteln beschrieben natürlich wünschenswert, wenn ein tiefes Netz wenigstens genauso gut ist wie ein weniger tiefes, gleich aufgebautes Netz. In der Praxis kommt jedoch das degeneration problem dazwischen, und ein Hauptgrund dafür ist, dass die Identitätsfunktion normalerweise nur schwer gelernt werden kann.

![Direktes Lernen der Zielfunktion  vs.  Lernen von Residuen](../images/residuals.png){ width=70% }

Mit der **Reformulierung des Lernprozesses** zu dem Lernen von Residuen wird dieses Problem nun jedoch gelöst. Wenn die Identitätsfunktion die optimale Funktion für die betrachteten layer ist, kann der solver die Gewichte von _F(x)_ relativ einfach gegen 0 steuern, da sie ohnehin wie oben besprochen bereits relativ nah um die 0 herum verteilt sind, sodass _F(x) = 0_ wird. Somit wird mit H(x) = 0 + x = x die gewünschte Identitätsfunktion erreicht. Die Eingabe in die layer wird einfach unverändert an hintere layer weitergereicht. Und wenn _H(x)_ nicht die Identiät ist, lernen die layer mit _F(x)_ eben alles nötige "was _x_ noch fehlt" um _H(x)_ zu approximieren.

Doch wie genau wird dies in einem ResNet umgesetzt und wie sieht ein ResNet überhaupt aus?

### Aufbau eines ResNets

![Aufbau eines ResNet-Blocks](../images/resnet_block.png){ width=55% }

Ein **ResNet** besteht aus einer Reihe von hintereinander geschalteten **ResNet-Blöcken** (s. Abbildung oben). Ein ResNet-Block umfasst typischerweise zwei bis drei normale hidden layer, das könnten beispielsweise zwei Convolutional layer mit anschließenden pooling und batch normalization layern sein wie im Beispiel unten. Es müssen mindestens zwei sein, da sie sich ansonsten nicht von normalen linearen Netzen unterscheiden würden. <span class="mark">Das besondere an ResNets sind jedoch die sogenannten 'shortcut-' oder **'skip-connections'**,</span> welche die Eingabe in die layer weiter nach vorne transportieren wo diese zu der Ausgabe aufaddiert werden, in der Regel noch bevor die Aktivierungsfunktion angewandt wird.

Formal gesehen ist ein ResNet Block _y_ also definiert als

<div class="center">_y = F(x, {W<sub>i</sub>}) + W<sub>s</sub>\ * x_,</div>

wobei _F(x ,{W<sub>i</sub>})_ die von den layern zu lernende Residualfunktion parametrisiert mit Gewichten _W<sub>i</sub>_ darstellt und _x_ die Eingabe in die layer bezeichnen. Somit approximiert der gesamte Block die Idealfunktion _H(x)_, also _y ≈ H(x)_.

![Beispiel für einen ResNet-Block mit convolutional layern](../images/1x1.png){ width=60% }

Da die Dimension der Eingabe nicht unbedingt der Dimension der Ausgabe der layer entsprechen muss, zum Beispiel weil convolutions und pooling angewandt wurden, muss die Eingabe _x_ in diesem Fall auf die Dimension der Ausgabe gebracht werden, dies wird durch die lineare Projektion _W<sub>s</sub>_ erreicht. In der Praxis werden dafür oft Padding-Methoden genutzt, jedoch haben sich **1x1 convolutions** als gängige Methode durchgesetzt.

Ansonsten besitzen die skip-connections bei klassischen ResNets keine weiteren Parameter die gelernt werden müssten, sodass durch Setzen der Gewichte _W<sub>i</sub>_ auf 0 der Block tatsächlich die Identitätsfunktion annähert.

Viele bekannte Deep Learning Bibliotheken wie Tensorflow und PyTorch enthalten oftmals vorimplementierte ResNet Architekturen. Möchte man jedoch ein ResNet von Grund auf selbst schreiben, ist hier mal eine Beispielimplementierung eines ResNet-Blocks in PyTorch vorgestellt. Diese Blöcke können dann hintereinander gesetzt werden, um ein ResNet zu bauen:

<!-- HTML generated using hilite.me --><div class="code" style="background: #f0f0f0; overflow:auto;width:auto;font-size:0.9em;border:solid gray;border-width:.1em .1em .1em .8em;padding:.2em .6em;"><table><tr><td><pre style="margin: 0; line-height: 125%"> 1
 2
 3
 4
 5
 6
 7
 8
 9
10
11
12
13
14
15
16
17</pre></td><td><pre style="margin: 0; line-height: 125%"><span style="color: #007020; font-weight: bold">class</span> <span style="color: #0e84b5; font-weight: bold">ResidualBlock</span>(nn<span style="color: #666666">.</span>Module):
    <span style="color: #007020; font-weight: bold">def</span> <span style="color: #06287e">__init__</span>(<span style="color: #007020">self</span>, in_channels, out_channels):
        <span style="color: #007020">super</span>()<span style="color: #666666">.</span>__init__()
        <span style="color: #007020">self</span><span style="color: #666666">.</span>in_channels, <span style="color: #007020">self</span><span style="color: #666666">.</span>out_channels <span style="color: #666666">=</span>  in_channels, out_channels
        <span style="color: #007020">self</span><span style="color: #666666">.</span>blocks <span style="color: #666666">=</span> nn<span style="color: #666666">.</span>Identity()
        <span style="color: #007020">self</span><span style="color: #666666">.</span>shortcut <span style="color: #666666">=</span> nn<span style="color: #666666">.</span>Identity()

    <span style="color: #007020; font-weight: bold">def</span> <span style="color: #06287e">forward</span>(<span style="color: #007020">self</span>, x):
        residual <span style="color: #666666">=</span> x
        <span style="color: #007020; font-weight: bold">if</span> <span style="color: #007020">self</span><span style="color: #666666">.</span>should_apply_shortcut: residual <span style="color: #666666">=</span> <span style="color: #007020">self</span><span style="color: #666666">.</span>shortcut(x)
        x <span style="color: #666666">=</span> <span style="color: #007020">self</span><span style="color: #666666">.</span>blocks(x)
        x <span style="color: #666666">+=</span> residual
        <span style="color: #007020; font-weight: bold">return</span> x

    <span style="color: #555555; font-weight: bold">@property</span>
    <span style="color: #007020; font-weight: bold">def</span> <span style="color: #06287e">should_apply_shortcut</span>(<span style="color: #007020">self</span>):
        <span style="color: #007020; font-weight: bold">return</span> <span style="color: #007020">self</span><span style="color: #666666">.</span>in_channels <span style="color: #666666">!=</span> <span style="color: #007020">self</span><span style="color: #666666">.</span>out_channels
</pre></td></tr></table></div>

Die folgende Abbildung stellt das durch das Original-Paper bekannt gewordene **152-layer ResNet** in den Vergleich zum prominenten VGG-19 Netz. Die gestrichelten skip-connections im rechten Bild kennzeichnen dabei 1x1 convolutions zur Anpassung der Dimensionen.

![VGG-19 vs ResNet-152](../images/vgg_resnets.png){ width=55% }

### Performances von ResNets

![Performance von ResNet<sup>1</sup>](../images/performance.png){ width=90% }

Und tatsächlich sind sehr tiefe ResNets ziemlich effizient trainierbar und erzielen gute Resultate. Wie in der obigen Abbildung zu erkennen <span class="mark">scheinen ResNets nicht vom degeneration problem betroffen zu sein.</span> Das Hinzufügen von mehr layern führt tatsächlich zu einem **niedrigeren Fehler und besserer Performance**, sowohl auf den Trainings- als auch auf den Testdaten. Tiefe ResNets erweisen sich als mindestens genauso gut und sogar besser als weniger tiefe, alle zusätzlichen layer nutzen scheinbar ihr Potential, die Performance zu verbessern.

![ImageNet competition: ResNet-152 überholt alle anderen](images/imagenet_competition2.png){ width=65% }

Wie in der Abbildung zu erkannen, gewannen ResNets mit dem berühmten ResNet-152 die **ImageNet comepetition** im Jahre 2015 mit einem Fehler von nur 3.57% und lösten damit das bis daher beste GoogLeNet mit einem Fehler von 6.7% und 22 layern deutlich ab. Interessanterweise hatte das ResNet-152 viel weniger Parameter als die anderen Wettbewerber und somit auch schneller trainierbar, war jedoch mit 152 layern sehr viel tiefer. Mehr dazu findet sich im übernächsten Kapitel zum Training von ResNets.

Seit dem konnten Architekturen von **über 1000 layern** erfolgreich trainiert werden, und im sehr theoretischen Umfeld auch Netze mit 2000+ layern, sie werden in der Praxis jedoch nicht wirklich eingesetzt, da sie eher unpraktikabel sind. Denn auch ResNets haben ihre Grenzen. Ab einem bestimmten Punkt läuft man nämlich auch Gefahr, die Trainingsdaten zu overfitten, sodass die Performance auch bei ResNets irgendwann abnimmt und mehr layer nicht mehr nützlich, sondern eher schädlich werden.

### Warum genau funktioniert das nochmal?

Zum einen sind die guten Performances von ResNets auf den Umstand zurückzuführen, dass das **Lernen von Residuen** scheinbar grundsätzlich einfacher zu sein scheint, als das direkte Approximieren einer Zielfunktion. Damit lassen sich auch sehr viel einfacher **Identitätsfunktionen** lernen, sodass es einem größeren Netz die Möglichkeit gibt, wenigstens genauso gut zu sein wie ein gleiches kleineres. <span class="mark">Layer, die der Performance nur schaden, können einfach übersprungen werden, indem die Eingabe in diese layer unverändert an hintere layer weitergereicht wird.</span> Dies geschieht auch ganz natürlich und dynamisch durch backpropagation, man braucht keinen zusätzlichen Hyperparameter für die Anzahl der layer hinzuzufügen.

Dadurch, dass gewisse layer einfach übersprungen werden könnten, ähnelt das Training von ResNets dem **Training eines ensembles**. Es erlaubt es, unterschiedliche Teile des Netzwerks zu unterschiedlichen Zeiten und Raten zu trainieren, abhängig davon, wie der Error im Netz zurückpropagiert wird. Somit können durch bestimmte Trainingsbeispiele auf natürliche Weise gezielt Teile des Netztes trainiert werden.

Da nur Residuen gelernt werden, geschieht im Prinzip lediglich ein **fine-tuning des Input** in einen ResNet Block. Er wird von jedem layer nur ein Stück weit angepasst, um näher an die zu lernende Idealfunktion zu kommen. Die erwartete Ausgabe muss somit nicht 'von scratch' generiert werden. Dies erklärt auch, warum das Hinzufügen von layern die Performance noch weiter erhöht: der Input wird mit jedem layer immer noch ein ganz kleines Stück verbessert - zumindest solange, bis man an Overfitting stößt.

### Training von ResNets

Schauen wir uns nun an, wie genau das Training von ResNets aussieht. Der Vorteil bei ResNets gegenüber anderen Netzwerk-Architekturen ist, dass ResNets zwar mehr layer besitzen, jedoch **sehr viel weniger Parameter**. Das end-to-end Training verkürzt sich dadurch mit normalen Verfahren oft trotz der höheren Anzahl an layern. 

Vergleichen wir zum Beispiel mal das ResNet-152 mit dem VGG-16 (ein grafischer Vergleich mit dem VGG-19 ist in "Aufbau eines ResNets" zu finden). Das VGG-16 besteht aus mehr als 143,6 Millionen Parametern, während das ResNet-152 nur 11,5 Millionen Parameter besitzt. Auch bei der Komplexität gibt es einen großen Unterschied. In dem Original-Paper zu ResNets geben die Autoren für das VGG-19 19,6 Milliarden FLOPs an, während ihr 34-layer Basisnetz mit nur 3,6 Milliard FLOPs auskommt, also nur 18% des VGG-19!

Das funktioniert auch deswegen besonders gut, da sich bei reinen ResNets nichts grundlegendes bei den foward- und backward-propagation Schritten ändert, die skip connections sind im Prinzip die einzige Änderung. Da diese in der Regel jedoch keine eigenen Parameter besitzen, <span class="mark">können ResNets also ganz normal mit herkömmlichen backpropagation Verfahren trainiert werden.</span> Das Training wird zusätzlich auch dadurch verbessert, dass die Gradienten über die skip connections auch viel besser zu vorigen layern fließen können. Dies hilft unter anderem auch bei dem vanishing/exploding gradients Problem.

### Andere Varianten von ResNets

Seit der Entwicklung von ResNets hat es mittlerweile viele Ansätze gegeben, die anfängliche Architektur zu verbessern und weitere, auf den einfachen ResNets aufbauende Netzwerkstrukturen zu entwickeln.

![Vergleich unterschiedlicher Anordnungen der layer in einem ResNet Block](../images/diff_resnetblocks.png){ width=90% }

Zum einen wurde untersucht, wie viele layer man am besten überspringt und welchen Einfluss die Reihenfolge der einzelnen Elemente in einem ResNet-Block auf die Performance hat<sup>2</sup>. Zum Beispiel könnte man den batch normalizaion layer nach der Addition mit dem 'skip'-Wert platzieren (b), oder die Aktivierungsfunktion schon vor der Addition anwenden (c), oder sie am Ende des Blockes komplett weglassen und stattdessen an den Anfang zu setzen (d). Die Autoren des entsprechenden papers fanden heraus, dass im Allgemeinen die Anordnung in **(e)** besonders bei sehr tiefen Netzen (~1000 layer) die besten Resultate liefert.

![Aufbau eines DenseBlocks](../images/dense_block.png){ width=70% }

Zum anderen wurden die Ideen in ResNets auch übernommen, um weitere vergleichbare Architekturen zu entwerfen. **DenseNets**<sup>4</sup> zum Beispiel führen statt nur einer skip-connection in einem layer Block Verbindungen von allem vorangehenden layer zu allen foldenden layern ein, daher der Name _dense_. Dabei werden viele dieser Blöcke wie bei ResNets auch hinterinander geschaltet. Der Vorteil ist, dass dadurch noch weniger Parameter gelernt werden müssen und die Gradienten noch besser propagiert werden können, was oft zu besseren Resultaten als bei ResNets führt.

Zwei weitere prominente Architekturen, die gewissermaßen auf ResNets aufbauen, sind **HighwayNets**<sup>5</sup> und **ResNeXt**<sup>6</sup>. Erstere führen für die skip-connections eigene Parameter ein, welche lernen, zu welchem Ausmaße die Eingaben durch die skip-connections oder durch den primären Weg weiterpropagiert werden. Die Struktur ähnelt gewissermaßen den gates in LSTMs. _ResNeXt_ Netze verfolgen dagegen den Ansatz, statt dem Hintereinanderschalten von einzelnen ResNet-Blöcken, stattdessen etliche solcher Blöcke parallel zu einem größeren, 'weiten' Block zu schalten, und dann diese größeren Blöcke hintereinander zu setzen. Beide Ansätze kommen mit ihren Vor- und Nachteilen und können bei bestimmten Problemen jeweils bessere Ergebnisse erzielen.


## Zusammenfassung

<span class="mark">**ResNets** erlauben es, sehr tiefe Netze zu trainieren.</span> Tiefere Netze ermöglichen es oftmals, effizienter komplexe Funktionen zu lernen und führen somit häufig zu besseren Ergebnissen. Da einfache Netze wegen des _degeneration problems_ nicht besonders tief sein können, formulieren ResNets das Lernproblem dazu um, in mehreren layer Blöcken **Residuen von Eingaben** zu einer gewünschten Idealfunktion zu lernen, anstatt letztere direkt mit dem Block zu approximieren. Das erlaubt es, relativ einfach die _Identitätsfunktion_ zu lernen, sodass das Hinzufügen von mehr layern die Performance nicht verschlechtert, sondern im Bestfall etwas mehr verbessert. Umgesetzt wird das ganze über **skip-connections**, also dem Weitergeben der Ausgabe eines layers an weiter hinten liegende layer.

Weil ResNets damit besonders erfolgreich waren, vor allem zur der Zeit als sie entwickelt wurden, wurden und werden sie häufig im Zusammenhang mit Bilderkennung genutzt, wo sie große Durchbrüche erzielen konnten. Mit der Zeit entstanden immer mehr Architekturen, die auf ResNets aufbauen, wie zum Beispiel _HighwayNets_, _DenseNets_ oder _ResNeXt_.

Doch auch ResNets haben ihre Grenzen, und man kommt ab einem gewissen Punkt zum _overfitting_ Problem. Daher sollte man in der Praxis stets die Architektur wählen, die sich am besten für ein gegebenes konkretes Problem eignet.


<br><br><br>
<br><br><br>

---


### Referenzen
<div class="references">
<sup>1</sup> Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (2016), **Deep Residual Learning for Image Recognition**, _IEEE_, 12. December 

<sup>2</sup> Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (2016), **Identity Mappings in Deep Residual Networks**. In: _Leibe B., Matas J., Sebe N., Welling M. (eds) Computer Vision_, ECCV 2016

<sup>3</sup> Mohammad Sadegh Ebrahimi, Hossein Karkeh Abadi (2018), **Study of Residual Networks for Image Recognition**, _arXiv:1805.00325_, 21. April

<sup>4</sup> Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger (2017), **Densely Connected Convolutional Networks**, _2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_, 2017, pp. 2261-2269

<sup>5</sup> Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber (2015), **Highway Networks**, _arXiv:1505.00387_, 03. November

<sup>6</sup> Saining Xie; Ross Girshick; Piotr Dollár; Zhuowen Tu; Kaiming He (2017), **Aggregated Residual Transformations for Deep Neural Networks**, _2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_, 2017, pp. 5987-5995
</div>


<br><br>
