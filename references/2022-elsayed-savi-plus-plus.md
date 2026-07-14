|     | SAVi++: |          | Towards |      | End-to-End |     | Object-Centric |     |
| --- | ------- | -------- | ------- | ---- | ---------- | --- | -------------- | --- |
|     |         | Learning |         | from | Real-World |     | Videos         |     |
GamaleldinF.Elsayed∗†,AravindhMahendran∗(cid:5),SjoerdvanSteenkiste∗(cid:5),
KlausGreff,MichaelC.Mozer&ThomasKipf∗
GoogleResearch
2202 ceD 32  ]VC.sc[  2v46770.6022:viXra
Abstract
Thevisualworldcanbeparsimoniouslycharacterizedintermsofdistinctentities
|     | with | sparse interactions. |     | Discovering | this compositional |     | structure | in dynamic |
| --- | ---- | -------------------- | --- | ----------- | ------------------ | --- | --------- | ---------- |
visualsceneshasprovenchallengingforend-to-endcomputervisionapproaches
unlessexplicitinstance-levelsupervisionisprovided.Slot-basedmodelsleveraging
motioncueshaverecentlyshowngreatpromiseinlearningtorepresent,segment,
andtrackobjectswithoutdirectsupervision,buttheystillfailtoscaletocomplex
|     | real-worldmulti-objectvideos. |     |     |     | Inanefforttobridgethisgap,wetakeinspiration |     |     |     |
| --- | ----------------------------- | --- | --- | --- | ------------------------------------------- | --- | --- | --- |
fromhumandevelopmentandhypothesizethatinformationaboutscenegeometry
|     | intheformofdepthsignalscanfacilitateobject-centriclearning. |     |     |     |     |     |     | Weintroduce |
| --- | ----------------------------------------------------------- | --- | --- | --- | --- | --- | --- | ----------- |
SAVi++,anobject-centricvideomodelwhichistrainedtopredictdepthsignals
|     | fromaslot-basedvideorepresentation. |     |     |     | Byfurtherleveragingbestpracticesfor |     |     |     |
| --- | ----------------------------------- | --- | --- | --- | ----------------------------------- | --- | --- | --- |
modelscaling,weareabletotrainSAVi++tosegmentcomplexdynamicscenes
|     | recorded | with | moving | cameras, | containing both | static | and moving | objects of |
| --- | -------- | ---- | ------ | -------- | --------------- | ------ | ---------- | ---------- |
diverseappearanceonnaturalisticbackgrounds,withouttheneedforsegmentation
|     | supervision. | Finally,wedemonstratethatbyusingsparsedepthsignalsobtained |     |     |     |     |     |     |
| --- | ------------ | ---------------------------------------------------------- | --- | --- | --- | --- | --- | --- |
fromLiDAR,SAVi++isabletolearnemergentobjectsegmentationandtracking
fromvideosinthereal-worldWaymoOpendataset.
|     | Projectpage: | https://slot-attention-video.github.io/savi++/ |     |     |     |     |     |     |
| --- | ------------ | ---------------------------------------------- | --- | --- | --- | --- | --- | --- |
Emergent segmentation
1 Introduction
time
| The | natural world | consists | of  | distinct |     |     |     |     |
| --- | ------------- | -------- | --- | -------- | --- | --- | --- | --- |
Optional conditioning signal
entities—people,dogs,cars,trees,etc.—
| and | its complexity | emerges | from | the |     |     |     |     |
| --- | -------------- | ------- | ---- | --- | --- | --- | --- | --- |
Init.
combined,mostlyindependent,actions
| oftheentities. |         | Thiscompositionalstruc- |           |     |     |     |     |     |
| -------------- | ------- | ----------------------- | --------- | --- | --- | --- | --- | --- |
| ture           | must be | appreciated             | topredict | fu- |     |     |     |     |
e.g. 1st-frame bounding boxes
turestatesoftheworldandtoeffectpar-
| ticular                        | outcomes. | People | have | an in-  |     |     | SAVi++ |     |
| ------------------------------ | --------- | ------ | ---- | ------- | --- | --- | ------ | --- |
| trinsicunderstandingofobjects: |           |        |      | objects |     |     |        |     |
havespatiotemporalcoherence,theyin-
|     |     |     |     |     | Figure1: EmergentsegmentationandtrackinginSAVi++. |     |     |     |
| --- | --- | --- | --- | --- | ------------------------------------------------- | --- | --- | --- |
teractwhenincloseproximity,andthey
∗Equaltechnicalcontribution.(cid:5)Alphabeticalorder.†Correspondenceto:gamaleldin@google.com
Authorcontributions:GFE,TKinitiatedandledtheproject.GFE,AM,SVS,TKdevelopedthemainmodel.
GFE,AM,TKdevelopedreal-worlddrivingdataandmodelinfrastructure.AMimplementeddataaugmentation.
GFEledablationstudy.SVSledtargetsignalsanalysesandmetricdesign.GFE,AM,SVS,TKranexperiments.
AM,SVS,KG,TKworkedonbaselines. KG,MCMprovidedadviceatallstagesandhelpedwithproject
scoping.TKdevelopedvisualizations.GFE,MCM,TKworkedonfiguredesign.Allauthorswrotethepaper.
36thConferenceonNeuralInformationProcessingSystems(NeurIPS2022).
t = 1
t = 3
t = 5

possesspersistent,latentcharacteristicsthatdeterminetheirbehavioroverextendedperiodsoftime
[31,55]. Justasobject-centricrepresentationsarecriticaltohumanunderstanding,theyhavethe
potentialinmachinelearningtogreatlyimprovesampleefficiency,robustness,visualreasoning,and
interpretabilityoflearningalgorithms[19,41]. Forexample, considerthechallengefacedbyan
autonomousvehicleoperatingindiversesurroundings(Figure1). Generalizationacrosssituations
requireslearningaboutrecurringentitieslikecars,trafficlights,andpedestrians,andtherulesthat
governinteractionsamongtheseentities.
Inhumanbrains, theabilitytoorganizeedgesandsurfacesintounitary, bounded, andpersisting
object representations develops through experience and/or maturation from infancy and without
explicitinstructionviaa‘coresystemofobjectrepresentation’[55],i.e.,aformofcognitiveinductive
bias. Indeeplearning,suchaninductivebiashasbeenproposedinslot-basedarchitectureswhich
segregate knowledge about individual objects into nonoverlapping but interchangeable pools of
neurons. Theresultingrepresentationalmodularitycanfacilitatecausalreasoningandprediction
fordownstreamtasks[19,53].
Agrandchallengeincomputervisionhasbeentodiscoverthecompositionalstructureofreal-world
dynamic visual scenes in an unsupervised fashion. By unsupervised, we mean no segmentation
informationisprovidedthatspecifieswhichpixelsbelongtogetheraspartofasingleobject. Initial
efforts focused on single-frame, synthetic RGB images [17, 18, 42, 60], but extending this work
tovideoandmorecomplexscenesprovedchallenging. Akeyinsighttofurtherprogresswasthe
realization that a color-intensity pixel array is not the only source of visual information readily
available,atleastnottohumanperceptualsystems. Thehumanperceptualsystemextractsmotion
anddepthcuesearlyintheprocessingstream[11–13,25,47]. Thesecuesarecorrelatedwithobject
identities,andcanthereforebootstraptheformationofobject-centricrepresentations[54].
TherecentlyintroducedSlotAttentionforVideo(SAVi)model[37]leveragedopticalflow(frame-
to-framemotion)asapredictiontargettoobtainobject-centricrepresentationsofdynamicscenes
involvingcomplex3Dscannedobjectsandreal-worldbackgrounds. However,motionprediction
aloneisinsufficienttolearnaboutthedistinctionbetweenstaticobjectsandthebackground. Further,
in real-world application domains such as self-driving cars, cameras themselves are subject to
movement,whichgloballyaffectsframe-to-framemotionasapredictionsignalinnon-trivialways.
In the present work, we describe an enhanced slot-based model for video, referred to as SAVi++
(Figure2),whichobtainsqualitativeimprovementsinobject-centricrepresentationsbyexploiting
depthsignalsreadilyavailablefromRGB-DcamerasandLiDARsensors. SAVi++isthefirstslot-
based,end-to-endtrainedmodelthatsuccessfullysegmentscomplexobjectsinnaturalistic,real-world
videosequenceswithoutusingdirectsegmentationortrackingsupervision.
Asummaryofourcontributionsisasfollows:
• WeintroduceSAVi++: anobject-centricslot-basedvideomodelthatmakesseveralkeyimprove-
mentstoSAVi[37]byutilizingdepthpredictionandbyadoptingbestpracticesformodelscaling
intermsofarchitecturedesignanddataaugmentation.
• On the multi-object video (MOVi) benchmark containing synthetic videos of high visual and
dynamiccomplexity[20],wefindthatSAVi++isabletohandlevideoscontainingcomplexshapes
and backgrounds, and a large number of objects per scene. Improving on SAVi, our approach
accommodatesbothstaticanddynamicobjectsandbothstaticandmovingcameras.
• Finally, we demonstrate that SAVi++ trained with sparse depth signals obtained from LiDAR
enablesemergentobjectdecompositionandtrackinginreal-worlddrivingvideosfromtheWaymo
Opendataset[56].
2 Relatedwork
Object-centric learning A growing body of research is addressing the problem of end-to-end
learningofobject-centricrepresentationsfromrawperceptualdatawithoutdirectsupervision. Slot-
basedneuralnetworkssuchasIODINE[18],MONet[5],andSlotAttention[42]relyonafactorized
latentspaceandindependentper-objectdecodersasinductivebiastoenableobjectdiscoveryina
simpleauto-encodingsetup. Architectureswithstrongerinductivebiasesusingfixedobjectsize,
presence,orpropagationpriorshavebeenexploredinworkssuchasSQAIR[39]andSCALOR[29],
butgenerallythesemethodshavefacedchallengesscalingtomorecomplexreal-worlddatawhen
2

relying on auto-encoding alone. Our work primarily builds on recent advances in object-centric
generative models for video sequences [30, 37, 60, 62, 70]. Different from our approach, these
methods have so far been unable to scale to complex real-world multi-object video data. An
alternativeclassofmethodsusingcontrastivelearningforobjectdiscovery[24,36,44,67],most
notablyGroupViT[67]andODIN[24],hasrecentlyachievedsomesuccessindiscoveringsemantic
groupingsinreal-worldimages.However,neitherGroupViTnorODINmodeldynamicsandtypically
failtoseparatesemanticallysimilarobjectinstancesincloseproximity. Inourwork,wefollowa
generativeapproach,butinsteadoftaskingthedecodertogeneratecomplexvisualRGBpixeldata,
weutilizedepthinformationtobootstrapobject-centriclearningwithoutdirectsupervision.
Objectdiscoveryindrivingscenes Arangeofrecentmethods[1,21,58,63]useamulti-stage
pipeline of 1) obtaining pseudo ground truth (PGT) segmentation or detection labels via some
heuristic, and 2) training a model in a supervised fashion on PGT labels. While this class of
methods achieves some success in discovering and tracking objects in real-world driving scenes,
it crucially hinges on the quality of the PGT labels, requiring carefully engineered task-specific
heuristicstoextractobjects. Earliermethodssolelyuseclusteringheuristicstoextractapproximate
segmentationmasksdirectlyfrommotiontrajectoriesformovingobjects[4,48]. Inourwork,we
insteaddemonstratethatobjectsegmentationandtrackingcanemergeinanend-to-endsettingon
complexreal-worlddatawithoutrelyingonPGTlabelgeneration.
Cross-modal learning For self-supervised object-centric learning from visual data, a range of
targetmodalitiesandtrainingsignalshavebeenexploredintheliterature. Byusingmotioncuesfrom
opticalflowaspredictiontargets,severalrecentmethods[37,68]wereabletoovercomelimitations
ofpurelyRGBpixel-levelgenerativemodels,whichfrequentlyfailedinthepresenceofcomplicated
textures[33]. However,thisadvantageisprimarilylimitedtodiscoveryofmovingobjects. Utilizing
depthtargetsfromasimulator[2]orfromsparseLiDAR[21,58,63],hasbeenexploredinaneffort
toovercometheselimitations. Differentfrompriorworksutilizingmulti-stagepipelinesandhand-
craftedheuristicsforextractionofpseudo-labelsfromLiDAR[21,58,63],wedirectlyutilizethe
(sparse)depthsignalastargetanddemonstratethatthiscanenableemergentobjectsegmentationand
trackingonreal-worlddrivingdatawithoutanyadditionalregularizersorpseudo-labelingtechniques.
Scalingstrategiesforvisionmodels Itiscommonpracticetoscalearchitecturalcapacitywith
datasetcomplexityandsize,whilemakinguseofstrongdataaugmentationwhenaddressingvarious
supervisedcomputervisiontasks[8,10,22,38]. Nonetheless,self-supervisedmethodsforend-to-
endobjectdiscoveryhaveprimarilybeenrelyingonoverlysimplisticandlow-capacitybackbone
architectures [18, 42, 70], likely due to the simplicity of datasets and tasks considered in prior
work. By scaling object-centric methods to larger, visually more complex datasets, we find that
utilizing stronger visual backbone architectures—in combination with data augmentation—can
providesubstantialbenefits. Forsimplisticdatasetswithlowervisualcomplexity(andsamenumber
ofexamples),wefoundanecdotalevidenceinpreliminaryexperimentsfortheoppositeeffect: both
architecturescalinganddataaugmentationcannegativelyaffectobjectdiscoveryperformance,likely
explainingwhypriorworkshavenotexploredthesestrategies.
Depthestimation Recentadvancesinsupervisedmonoculardepthestimation(seeMingetal.[46]
forareview)couldbecombinedwithourmethodinfuturework,forinstanceusingordinalregression
losses[14],transformerarchitectures[52],ormorecomplexinstance-wisedecoderarchitectures[64].
3 Methods
WebeginbyprovidingabriefintroductiontoSlotAttentionforVideo(SAVi),whichisthestarting
pointforourexploration.WithSAVi++,weintroduceseveralsimpleyetcrucialimprovements,which
allowustobridgethegaptocomplexreal-worlddata. OurframeworkissummarizedinFigure2.
3.1 Background
SlotAttentionforVideo(orSAVi)isarecentstate-of-the-artarchitectureforlearningobject-centric
representationsfromvideowithminimalsupervision.Webrieflyhighlightsomeofitskeycomponents
belowandreferthereaderforcompletedetailstoKipfetal.[37].
SAVicanbeviewedasanautoregressiveencoder-decodervideomodelwithastructuredlatentstate
composedofK objectslots. Atagiventime-step,anencoderfirstencodestheobservedvideoframe
3

...
Emergent segmentation
CNE
time
||...||2
Depth prediction Ground-truth
sparse depth
Augmented frames
SAVi++
encoder Object
Slot
slots
decoder
CED
CED
tim
e
Video frames
Figure2: SAVi++isanobject-centricvideomodelbasedonSlotAttentionforVideo[37],which
encodesavideointoasetoftemporally-consistentlatentvariables(objectslots). Inputframesand
predictiontargetsareaugmentedusingrandomcropaugmentations. Augmentedframesarepassed
through the improved SAVi++ encoder and mapped onto object slots using an attention mecha-
nism[42]. Slotsareupdatedrecurrentlyforeachframeandsubsequentlydecodedindependently
intoadepthmapandper-slotalphamasks. SAVi++istrainedusing(sparse)depthtargets,leadingto
emergenceoftemporally-consistentobjectsegmentationinthedecodedalphamasks.
to yield high-level image features that are useful for learning about objects. This is followed by
SlotAttention[42](the‘corrector’),whichupdatestheslotsusingthesefeaturesandencourages
individualslotstospecializetodifferentpartsoftheobservation. Thecontentofeachslotisdecoded
separatelyusingadecoder,whichadditionallyoutputsapixel-levelalphamasktoindicatehowthe
decodedvaluesforeachslotshouldbecombined. Together,themaskanddecodedslotsdetermine
theoutputofthemodelatthecurrenttime-stepfromwhichalossiscomputed,e.g.,totrainthemodel
topredictframe-to-framemotion(opticalflow)forthisframe. Slotsforthenexttime-step(forthe
correctortoupdate)areobtainedbyapplyingapredictor,whichcanmodelinteractionsbetweenslots
andlearnaboutobjectdynamicstopredicttheirfuturestate.
Inadditiontoopticalflowprediction,SAViintroducesconditioningthathelpsreduceuncertainty
aboutthepart-wholedivisionintoobjectsbypointingthemodeltospecificlocations. Indeed,in
theabsenceofaspecificdownstreamtask,scenedecompositioncanbeambiguousandproviding
additionalinformationasaconditioningsignalmayhelpalleviatethis. Theconditioningtakesplace
viatheslotinitializer,whichinitializestheslotsusedintheinitialvideoframe. Theinitializationmay
belearnedinanunconditionalsetting(i.e.,learntheinitialslotstates)orobtainedbyconditioning
theinitialstateonhigh-levelcuessuchasboundingboxesofobjectsofinterestinthefirstvideo
frame. ThisdirectionofattentionorinputconditioninghelpedSAVitosucceedindecomposingmore
complexvisualscenes.
3.2 SAVi++
AsSAVireliesonopticalflowpredictionasitsmaintrainingsignalforobjectdiscovery,itsapplication
isprimarilylimitedtosettingswhereallobjectsinascenehaveindependentmotion. Inaddition,
SAVistruggledtogeneralizetosceneswithamovingcamera, eventhoughtheopticalflowfield
encodesinformationabout(static)scenegeometryinthiscase.
Here,weidentifytwokeydirectionsforimprovingSAViandbridgingitscapabilitiestoreal-world
videodata,whilepreservingitscorefoundationforlearningobjectrepresentationsfromvideo: (1)
exploitingdepthasapredictionsignal,whichisreadilyavailableinmanyreal-worldsettings,and(2)
utilizingmodelscalingstrategiesintermsofencoderimprovementsanddataaugmentation,which,
despitebeingcommonly usedforclassicvisionproblems, aregenerally underutilized forobject-
centriclearning. Ourimprovedapproach,calledSAVi++,successfullysegmentscomplexobjectsin
naturalistic,real-worldvideosequenceswithoutusingdirectsegmentationortrackingsupervision.
Exploitingdepthinformation Trainingobject-centricmodelssolelyusingRGBimageorvideo
framereconstructionproveschallenginginthepresenceofcomplexvisualtextures,frequentlyleading
tofailuremodessuchasclusteringbycolororintoobject-agnosticspatialregions[18,21]. InSAVi,
opticalflowwasproposedasapredictionsignaltomitigatethisissue,whilestilloperatingonvisual
RGBinputs[37]. However,relyingsolelyonopticalflowasapredictiontargetforlearningabout
objectshasacleardisadvantage: staticobjects,whichmakeupthevastmajorityofvisualentitieswe
encounteronadailybasis,arenotcapturedinthissignalunlesstheobserverortheentiresceneisin
4

Video Flow Depth
C-iVOM
D-iVOM
E-iVOM
Video Sparse depth
(a)MOVidatasets. (b)WaymoOpendataset.
erutxetxelpmoC stcejbognivoM
stcejbocitatS
aremacgnivoM )niart(semarf#
MOVi-C (cid:51) (cid:51) (cid:55) (cid:55) 234k
MOVi-D (cid:51) (cid:51) (cid:51) (cid:55) 234k
MOVi-E (cid:51) (cid:51) (cid:51) (cid:51) 234k
Waymo (cid:51) (cid:51) (cid:51) (cid:51) 159.6k
Open
(c)Datasetdetailsandstatistics.
Figure3: WeconsiderthreesyntheticMulti-ObjectVideo(MOVi)datasets[20]andthelarge-scale
real-worlddrivingdatasetWaymoOpen[56]. Alldatasetscontaincomplextexturesandmoving
objects. TheMOVidatasetsincreaseincomplexityfromMOVi-C(movingobjectsonly)overMOVi-
D(+staticobjects)toMOVi-E(+movingcameras). WaymoOpencontainsallthesecharacterisics.
motion. Asaconsequence,SAVifailstorepresentobjectsthatareatrest,andsimilarlystruggleswith
scenesobservedfromamovingcamera,asopticalflowcanprovechallengingtomodelinthiscase.
Here,weexploredepthasatargetsignal,usedinconjunctionwithfloworeveninisolation. Depth
estimationhasreceivedlittleattentioninslot-basedmodels,yetdoesnotsufferfromthelimitationof
opticalflowindatasetswithstaticobjectsandcameramovement.Wethushypothesizethatdepthmay
greatlybenefitobtainingemergentobjectdecompositionsofcomplexvideos. Intermsofpractical
applicability,wenotehowdepthisareadily-availablesignalinmanyreal-worldsettingsthankstothe
prevalenceofRGB-DcamerasandLiDARinsettingslikeself-drivingcars[56]. Evenintheabsence
ofdepthsensingcapabilities,thissignalcanbecheaplyestimatedfrommulti-camerasystems[40].
Inourimplementation,werepresentthedepthsignalinimagespace,whichweencodeusingalog
transformationlog(1+d), wheredisthedistanceofapixeltothecamera(seeFigure3a). This
log-transformputsastrongeremphasisonclose-byobjectsand—inearlyexperiments—wefound
this form of normalization crucial for reliably training object-centric models using depth targets.
SAVi++isthentrainedtominimizethesquareddifferencebetweenthedecoderoutputandthistarget
signal. Incaseofmultipleavailabletargets,suchasdepthandflow,weconcatenatethetargetimages
alongthechanneldimensionandpredictthemusinganotherwiseunchangedmodel.
ForsparsetargetssuchasdepthobtainedfromLiDAR,weignoreanypointsintheimagespacefor
whichnosignalispresentinthecomputationoftheloss. ForLiDARspecifically,weobtainthex,y,
zcoordinatesofalltheLiDARpointsintheself-drivingcar(SDC)worldandcomputethedistanceof
eachofthepointsfromtheLiDARsensor. WethenusethecameraandLiDARcalibrationparameters
toprojecttheLiDARpointdistancesfromtheSDCdomaintothecameraframe. Thisprojection
representsaverysparseapproximationoftheground-truthdepthsignal(Figure3b).
Scaling strategies Visual complexity present in real-world videos necessitates a different class
ofencodersthanthoseusedforsimplesyntheticdatasets. Inspiredbysuccessfulvisualbackbone
architecturesforset-basedsupervisedobjectdetectionmodels[7,32],weuseamorecapableencoder
thatutilizestheResNet34[22]architecturefollowedbyatransformerencoder[61](with4layers,
unlessotherwisementioned). Toavoidcomputationofbatchand/ortemporalstatistics,wereplacethe
typicalbatchnormalizationinResNet34withgroupnormalization[65].Weuseastride1convolution
andusenomax-poolingintheResNetrootblock. Thisresultsinanoverallbackbonestrideof8(as
opposed32),whichwasfoundtobeimportantforretainingobjectdecompositioncapabilities. Please
seetheappendixforfurtherarchitecturaldetails.
Drawinginspirationfromtrainingschemescommonlyusedforreal-worldvisionmodels[57],we
furtherapplyInception-stylecroppingasdataaugmentation. Inparticular,werandomlycroparegion
ofeachframewithaspectratio∈[0.75,1.33]suchthatenoughoftheframeisretainedaftercropping.
The same crop is applied consistently across all frames and the resulting video is resized to the
originalresolution. Flowfieldsanddepthmapsareadjustedaccordinglytokeepthemaccurateand
spatiallyalignedwiththevideoframe.
5

4 Experiments
Thegoalofourexperimentalevaluationistwofold: 1)onsyntheticvideodataofvaryingcomplexity
we would like to analyze the potential advantages of utilizing a depth signal and model scaling
strategies for learning emergent segmentation and tracking, and 2) we would like to investigate
whethertheseimprovementsenablebridgingthegaptocomplexreal-worldvideodata.
Section4.1coversbothqualitativeandquantitativecomparisonsofSAVi++againstbaselineson
thesyntheticMOVidatasets. InSection4.2,weperformanablationstudyonSAVi++. Finally,in
Section4.3wedemonstrateandanalyzeresultsforaSAVi++modelappliedtoreal-worlddriving
videosfromtheWaymoOpen[56]dataset.
Datasets Asbasisforourexperiments,weusevideosofdifferentsceneandcameracomplexities
(Figure3c). WeusethreesyntheticMulti-ObjectVideo(MOVi)datasets(Figure3a)introducedin
Kubric[20],whicharecreatedbysimulatingrigidbodydynamics. Wenarrowourinvestigationto
MOVidatasetswithcomplexnaturalisticbackgroundsand3D-scannedeverydayobjects(variants
C,D,andE).MOVi-Cisgeneratedusingastaticcamera,andallobjects(max.10)areinitialized
tomoveindependently. MOVi-Dintroducesmoreobjects,someofwhicharedynamic(1-3)and
themajorityrestsstaticallyinthescene(10-20). Finally,MOVi-Eintroducesrandom,linearcamera
movement. Eachvideocontains24framessampledat12framespersecond(fps).
WealsotrainandevaluateSAVi++inareal-worlddrivingsettingusingtheWaymoOpendataset
(Figure 3b). Waymo Open is comprised of high resolution video data of 1280×1920 original
resolutionfromamulti-camerasystemcollectedbyWaymovehicles[56]. Thedatasetconsistsof
798trainand202validationscenesof20svideoeach,sampledat10fps. Wesubsamplethedataset
at5fpsbothfortrainingandvalidation. ThedatasetalsoincludesLiDARsignalsthatweuseto
computesparsedepthmapsasdiscussedinSection3.
Trainingsetup Forallourexperiments,unlessstatedotherwise,weresizeframestoaheightof128
pixelswhilekeepingtheaspectratiofixed,resultingina128×128resolutionforMOVidatasets,and
aresolutionof128×192forWaymoOpen. WetrainSAVi++for500kstepsonTensorProcessing
Unit(TPU)acceleratorswithabatchsizeof64usingAdam[35].
Wetrainonrandomlysampledsub-sequencesofonly6framesusing24slotsforMOViand11slots
forWaymoOpen. Seeappendixforfurthertrainingdetailsandhyperparameters.
4.1 SAVi++improvesobject-centriclearningoncomplexsyntheticvideodata
WeinvestigatewhetherthekeychangesintroducedtoSAVi[37], whichconstituteourimproved
SAVi++model,allowustoovercomelimitationsofSAViandaddressthemostchallengingsynthetic
multi-objectvideo(MOVi)benchmarksintroducedinKubric[20].
Setup We train all models independently on each dataset variant. Both SAVi and SAVi++ are
trainedinaconditionalsettingwhereweinitializeslotsusingground-truthboundingboxinformation
inthefirstframe. Wereportthesamesegmentationmetricsasinpriorwork,i.e.ForegroundAdjusted
RandIndex(FG-ARI) [27,51]andMeanIntersectionoverUnion(mIoU).FG-ARIisapermutation-
invariantclusteringsimilaritymetricfrequentlyusedforevaluatingscenedecompositionquality. It
comparesdiscoveredsegmentationmaskswithground-truthmaskswhileignoringanypixelsthat
belongtothebackground. Itissensitivetotemporalconsistencyofmasks,butinsensitivetotheir
ordering. The mIoU metric is a standard segmentation metric, here adapted for video as in [6].
Wenotethatthisimplementationissensitivetothecorrectorderingofmasks,i.e.italsomeasures
whethermodelsusedtheconditioningsignal(here,first-frameboundingboxes)correctly.
Baselines BesidescomparingtoSAVi[37],themostrepresentativepriormethodforthetaskwe
areinterestedin,wecompareagainstarangeofbaselinesaimedatestablishingthedifficultyofthe
unsupervised,boundingbox-conditionedvideoobjectsegmentationtask: 1)aboundingboxcopy
(BBoxcopy)baseline,whichsimplyrepeatsthefirst-frameboxesthroughoutthevideo,2)alearned
BBoxpropagationbaselinethatdoesnotreceivevisualinputs,totestforeasilyexploitablebiasesin
thedatasets,3)k-Meansclusteringbaselines,thatclustertheflowand/ordepthsignalacrossthevideo
sequence(initializedusingtheground-truthobjectcentersinthefirstframe),and4)alabelpropagation
baseline,thatusesvisualfeaturestopropagatetheinitialboxes(renderedasrectangularmasks)across
thevideo,basedonContrastiveRandomWalks(CRW)[28]. Seeappendixforfurtherdetails.
6

Table1: MOViresultsintermsofmeanscore±standarderror(5seeds)fromevaluatingSAVi++
andbaselinemodelsonvalidationsetvideosequencesofincreasedlength(24frames). *: weusethe
officialimplementationofCRW[28],whichdoesnotreportFG-ARI.
mIoU↑(%) FG-ARI↑(%)
Model MOVi-C MOVi-D MOVi-E MOVi-C MOVi-D MOVi-E
BBoxcopy 12.3 42.8 32.9 11.8 68.0 54.7
BBoxpropagation 22.9±0.1 26.7±0.8 24.1±1.1 9.6±0.5 24.9±3.7 18.4±3.9
K-Means(depth) 7.1±0.3 6.0±0.4 5.4±0.3 26.3±1.0 30.9±0.7 32.2±0.6
K-Means(flow) 10.7±0.5 7.4±0.4 6.0±0.3 26.5±1.0 30.9±0.8 33.1±0.7
K-Means(flow+depth) 10.6±0.6 6.7±0.4 5.3±0.3 26.6±1.0 35.9±1.0 34.8±0.7
CRW[28] 27.8±0.2 45.3±0.0 47.5±0.1 * * *
SAVi[37] 43.1±0.7 22.7±7.5 30.7±4.9 77.6±0.7 59.6±6.7 55.3±5.8
SAVi++(ours) 45.2±0.1 48.3±0.5 47.1±1.3 81.9±0.2 86.0±0.3 84.1±0.9
Results QuantitativeresultscanbeseeninTable1andqualitativeresultsonMOVi-EinFigure4a.
TheBBoxcopymethodservesasatrivialbaselinewhichalearning-basedapproachshouldoutperform.
WhiletheoriginalSAVimodeldoessoonMOVi-C,itclearlyfailstomodelthemorecomplexMOVi-
Dand-Edatasets. TheBBoxcopybaselinesis—perhapsunsurprisingly—strongestonMOVi-D,
wheremostobjectsarestatic. SAVi++outperformsthisbaselineonalldatasets,indicatingthatit
learnsnon-trivialsegmentationandtrackingcapabilities. Indeed,thisadvantagedoesnotsolelycome
fromfittingcertainbiasesinthedatasets,asalearnedBBoxpropagationbaseline(usingthesame
predictorasinSAVi++)thatdoesnotreceivevisualinput,failstogeneralizetounseenevaluation
videos. ItisworthnotingthatneitheroftheMOVitaskscaneasilybesolvedbysimplyclusteringthe
targetsignals,astheresultsforthek-Meansbaselinesdemonstrate.
Compared to CRW it can be seen how SAVi++ yields markedly better mIoU on MOVi-C and
D, while performance on MOVi-E is similar. Note that, unlike SAVi++, CRW is merely capable
of propagating pixel-level annotations across frames in a video and does not by itself produce
instance-levelobjectsegmentationsorcorrespondingobject-representationsthatcouldbeusedfor
down-streamtasks. Finally,comparingSAVi++andSAVidirectly,weseethatSAVi++overcomes
theprimarylimitationsofSAViontheharderMOVi-Dand-Edatasets,bothquantatively(Table1)
andqualitatively(Figure4a),whilealsoimprovingperformanceonMOVi-C.
Discussion It is evident that a small number of critical changes to SAVi [37], namely utilizing
depthtargets,astrongerarchitecture,anddataaugmentation,canhavedramaticconsequencesonthe
abilityofthisslot-basedmodeltolearnemergentobjectsegmentationandtrackingincomplexvideo
sequences. ThedifferencebetweenSAVi++andSAViisespeciallyevidentforthemorecomplex
datasetsinourstudy(e.g.,improvingthemIoUscoreonMOVi-Efrom30.7%to47.1%;seealso
Figure4a). TheseresultsdemonstratethatSAVi++isbettersuitedforvariousdatacomplexitiesin
termsofobjectdynamicsandcameramovement,whicharelikelytoexistinreal-worlddata.
4.2 Ablationstudy
In this section, we report results of an ablation study to gauge the contribution of the different
componentsofSAVi++. ThethreemainingredientsofSAVi++are1)theuseofdepthastraining
target, 2)theextracapacityaddedtoSAVibyincludingatransformerencoder, and3)theuseof
dataaugmentation. Figure4bshowsasystematicablationofeachofthosecomponents. Removing
thetransformerencoderreducesobjectsegmentationquality,yetthedegradationinperformanceis
relativelylimited. WhiledataaugmentationonlyhasamildeffectonthesimplerMOVi-Cdataset,it
makesasubstantialdifferenceonthemorechallengingdatasets,MOVi-DandE.Finally,removing
depthtargetsreducesperformancefurtherandisparticularlycatastrophiconMOVi-E.
Infact, wefindthattrainingsolelyusingdepthtargetswithoutrelyingonpredictingopticalflow
aswell(seew/oFlowinFigure4b)stillallowsthemodeltoaccuratelysegmentandtrackobjects,
especiallyonthemorecomplexMOVi-DandEdatasets. ThisresultisparticularlystrongonMOVi-E
where jointly predicting optical flow presents a difficult task for scenes with camera movement.
Further,trainingondepthtargetswasverycrucialtoobtaingoodperformanceonthemostcomplex
7

Input frame SAVi SAVi++ Ground truth 0.5
0.4
0.3
0.2
0.1
0.0
MOVi-C MOVi-D MOVi-E
(a)MOVi-Equalitativeresults.
UoIm
SAVi++
w/o Trans.
w/o Trans. & Aug.
w/o Trans. & Aug. & Depth
w/o Flow
w/o Depth
(b)Ablationstudy
Figure4:Left:QualitativeresultsofSAVi++comparedtoSAVi[37]onthesyntheticMOVi-Edataset
withcameramotion. Right: SAVi++ablationstudyonMOVi-C,D,andE.Barsreflectvalidation
setmIoU(mean±standarderrorfor5seeds). Weablate: 1)thetransformerencoder(w/oTrans.),
2)dataaugmentation(w/oTrans. &Aug.),and3)depthtargets(w/oTrans. &Aug. &Depth). We
furtherreportresultsfortrainingwithoutflow,whileonlyusingdepthtargets(w/oFlow).
Input frame SIMONe SIMONe + depth SAVi++ (unconditional) SAVi++ (conditional)
Figure5: QualitativesegmentationcomparisonontheWaymoOpenvalidationset. Naiveapplication
ofaSIMONe[30]baselinemodeltothisdatasetresultsinfailure,whileadaptingSIMONetopredict
(sparse)depthmapsyieldsrough(butfrequentlymisaligned)segmentationmasks. SAVi++generally
produceshighlyaccuratesegmentationmasks,whileitsunconditionalresultsarepromising. Here,
wehidemasksthatoccupymorethan1300pixelsonaverageperframetoeaseinterpretability.
syntheticdataMOVi-EasdemonstratedwiththelargedropinmIoUwhenablatingdepthandrelying
onlyonopticalflowtotrainthemodel.
4.3 SAVi++enablesemergentsegmentationonreal-worlddrivingdata
Intheprevioussection,wefoundthatsolelyusingdepthasatrainingtargetcanbesufficienttolearn
emergentobjectsegmentationandtracking. Thisfindingprovidesastrongmotivationforscaling
thisclassofmethodstoreal-worlddata,wheretheavailabilityofopticalflowreliesonapproximate
andpotentiallyinaccurateflowestimationmethods,whereasdepthcanbeaccuratelymeasuredusing
technologieslikeLiDAR.Toinvestigatethispossibility,weusetheWaymoOpendataset[56],which
includesvideosobtainedfromcamerasmountedoncarsinvarioustrafficenvironments.
Setup Toobtainadepthsignal,weproject3DLiDARpointsintothecameraframe,resultingin
averysparsedepthimageforeachtimestep(seeFigure3bforexamples). Weexcludepixelsthat
do not have a valid LiDAR point when computing the L2 loss in image space. We train SAVi++
with11slotson6framesandevaluatethemodelonsequencesof10frames. Duetotheabsenceof
ground-truthsegmentationlabelsinWaymoOpen,wequantitativelymeasureperformancecompared
toground-truthboundingboxesusingthreemetrics. TheCenter-of-Mass(CoM)distancemeasures
theaverageEuclideandistancebetweenthecentroidofthepredictedsegmentationmasksandthe
centers of the ground-truth bounding boxes. We report the centroid distance normalized by the
maximumachievabledistanceinthevideoframe. Additionally,weseparatelymeasurethefraction
8

| Conditioning |     | t = 0 | t = 5 |     | t = 10 | t = 15 | t = 20 | t = 25 |
| ------------ | --- | ----- | ----- | --- | ------ | ------ | ------ | ------ |
Figure6: WaymoOpenqualitativeresultsofSAVi++(conditional)overlongsequences.
ofcaseswhereanysortofsegmentispredictedwhenavalidground-truthboxexists,denotedas
bounding box recall (B. Recall). The Bounding Box mIoU (B. mIoU) is analog to mIoU using
predictedandground-truthboundingboxes. TheformerareobtainedbytrainingareadoutMLPto
predictboundingboxesfromtheslotrepresentations. Seeappendixforfurtherdetails.
Baselines Wequantitativelycomparetothesubsetofpreviousbaselinesthatworkwith(sparse)
depth. Further, wereportqualitativeresultsforSAVi++intheunconditionalsetting, i.e.without
providingfirst-frameboundingboxestothemodeltoinitializeslots,andcomparetoSIMONe[30]
asarepresentativeobject-centricvideomodelbaselinefromtheliterature.
Results Quantitative results can be seen in Table 2 and qualitative results in Figures 5–6. We
findthatSAVi++markedlyoutperformstheBBoxcopyandpropagationbaselines,aswellasthe
clusteringbaselineintermsofobjecttracking. Further,theboundingboxrecallishighindicating
thatvalidobjectsarerarelyignored. ThequalitativeresultsinFigures5–6evenbetterreflectthe
significanceofSAVi++’sperformanceaswellasitspotentialutilityforobject-centricrepresentation
learningfromreal-worldvideos(forSAVi++resultsdividedperobjectcategoryseeTable4).
OurresultsusingsparsedepthtargetssuggestthatSAVi++doesnotneedcomplete(i.e.dense)depth
supervision. Toinvestigatehowaccuratethissignalneedstobe,weexploredthedegreeofsensitivity
ofSAVi++tonoiseinthedepthsignal. WetrainedSAVi++withnoisydepthtargetsbyapplying
additiveGaussiannoisetotheground-truthsparseLiDARdepthsignalswithstandarddeviationsof
10cm,20cmand40cm. WefoundthatSAVi++wasabletoretainitsemergenttrackingperformance
evenatthehighestconsiderednoisescaleof40cm(seeTable5inAppendix).
| We additionally | experimented |     | with | removing |         |                                |     |     |
| --------------- | ------------ | --- | ---- | -------- | ------- | ------------------------------ | --- | --- |
|                 |              |     |      |          | Table2: | WaymoOpenresults(mean±standard |     |     |
theboundingboxconditioninginSAVi++inthe error in %, 3 seeds) from evaluating models on
initialframe. Removingthisconditioningsig- sequences of 10 frames. SAVi++ HR is a vari-
| nal and | using a learned | initialization |     | together |     |     |     |     |
| ------- | --------------- | -------------- | --- | -------- | --- | --- | --- | --- |
anttrainedonhigher-resolution(256×384)video
| with a simplified |     | encoder | also yielded | good |     |     |     |     |
| ----------------- | --- | ------- | ------------ | ---- | --- | --- | --- | --- |
frames.
| object decompositions |            | (see     | SAVi++ | (uncondi-   |     |     |     |     |
| --------------------- | ---------- | -------- | ------ | ----------- | --- | --- | --- | --- |
| tional) in            | Figure 5). | Compared | to     | using plain |     |     |     |     |
(%)
SIMONe[30],weobservethatSAVi++(uncon- Model CoM↓ B.mIoU↑ B.Recall↑
| ditional)performsmarkedlybetter.Interestingly, |     |     |     |     |          | 5.0 | 44.3 |     |
| ---------------------------------------------- | --- | --- | --- | --- | -------- | --- | ---- | --- |
|                                                |     |     |     |     | BBoxCopy |     |      | 100 |
modifying the non-autoregressive SIMONe BBoxProp. 5.1±0.1 38.5±0.5 100
baselinesimilartoSAVi++bypredictingsparse K-Means(depth) 13.0±0.1 – 100
depth instead of RGB also showed improve- SAVi(RGB) 21.5±1.8 7.9±0.9 95.8±2.7
|     |     |     |     |     | SAVi(depth) | 24.7±0.7 | 10.3±2.4 | 97.4±0.6 |
| --- | --- | --- | --- | --- | ----------- | -------- | -------- | -------- |
ment in object emergence. This gives further SAVi++ 4.4±0.2 49.7±0.7 96.5±0.7
evidencethatusingdepthissuitableforlearning
|     |     |     |     |     | SAVi++HR | 3.9±0.1 | 51.9±0.4 | 96.2±0.4 |
| --- | --- | --- | --- | --- | -------- | ------- | -------- | -------- |
object-centricrepresentationsfromrealvideos.
Quantitatively,SAVi++achievesaCoMdistance Supervised 1.1±0.0 67.6±0.6
| of 6.9±0.5 | while | SIMONe | (with | depth | loss) |     |     |     |
| ---------- | ----- | ------ | ----- | ----- | ----- | --- | --- | --- |
achieves7.4±0.21overasequenceof12framesattesttime,evaluatedusingHungarianmatching.
WeshowqualitativeresultsforlongersequencesinFigure6andinvideo-formatinthesupplementary
material. ItisworthnotingthatSAVi++wasonlytrainedon6framesanddidnotreceiveanytracking
supervision. Interestingly,wefindthatobjectsareoftenconsistentlytrackeduntilthemomentthey
leavethescene. Atthisstage,slotsarefreedupagainandtendtobindtopreviouslyunexplained
ornewobjects. Thisbehaviourindicatesthatourreportedtrackingmetricsareanunderestimation
ofthecapabilitiesofthemodel,assuchre-bindingisnotaccountedfor. Itis,however,conceivable
1Thesebaselineresultsareimprovedcomparedtoanearlierversionofthepaperbyusingexactlythesame
depthtargettransformationasforSAVi++.
9

thatre-bindingeventscouldbeidentifiedpost-hocifoneweretousetherepresentationslearnedby
SAVi++fordownstreamtasks,whichisaninterestingavenueforfuturework.
4.4 Limitations
WithSAVi++,wedemonstratedthefirstproofofconceptthatanemergentobject-centricdecomposi-
tionofreal-worldcomplexvideosispossiblewithanend-to-endslot-basedapproach. Yet,thereis
stillalotofroomforimprovement.
Relianceonconditioning Wefocusedourexplorationontheconditionalsetupwhereweprovided
cuesintheformofboundingboxesofobjectsinthefirstframe. Althoughtheuseofsuch“object
hints”maysharesomesimilaritytohowhumanvisualattention(andhowhumansparseavisual
scene)canbedirectedviaexternalsignals(e.g.,viagesturessuchaspointing),itultimatelylimitsthe
practicalapplicabilityofourapproach. PreliminaryresultswithunconditionalSAVi++suggestthat
thisinformationmaynotbestrictlynecessaryandcouldberemovedinfutureresearch.
Relianceonground-truthtargetsignals Inasimilarvein,therelianceofSAVi++onground-truth
targetsignalsfortrainingisalimitationthatmayaffectitspracticalapplicability. Fortunately,LiDAR
sensorsfordepthestimationarereadilyavailableinmanyapplicationdomains(suchasinrobotics
andself-driving),andthereisalsoarichliteratureonmonoculardepthestimation. Whileestimated,
depth(orflow)areexpectedtobenoisiercomparedtothesignalsconsideredinourexperiments,our
experimentwith“noisydepth”offersaninitialsignthatthismaynotaffectperformancemuch.
Gaptovideosrecordedinthewild ItisalsoimportanttopointoutthatalthoughWaymoOpen
offers a challenging real-world benchmark for learning about objects, its videos are relatively
structuredcomparedtoreal-worldvideosrecorded“inthewild”,andespeciallyheavyoncars,roads,
traffic signs, pedestrians, etc. Other datasets, such as DAVIS [49] or Kinetics [34] offer greater
complexityinthatregardanditisforeseeablethatfurtherdevelopmentofSAVi++willbeneeded
totrulysupportthese. AnexampleofthisisthatobjectsinWaymoOpenusuallydonotre-appear,
whichisanaspectthatiscurrentlynotexplicitlymodeledinSAVi++(e.g.toensurethatthesame
objectisre-capturedbythesameslot). Moregenerally,thereissubstantialheadroomtoimprovethe
modelingofdisappearingandreappearingobjectsinfuturework,suchasbyexplicitlymodeling
objectpresence[39],orbyexplicitlyattendingtopastlatentstates[69].
Gaptosupervisedapproaches Finally,wenotehowbothintheconditionalandtheunconditional
setting,thesegmentationandtrackingperformance,thoughimpressivegiventheminimalamountof
supervisionthemodelreceives,stillqualitativelylagsbehindsupervisedapproaches. Improvingon
thetemporalconsistencyofobjecttracks,especiallyintheunconditionalsetting,isanotherpromising
directionforfuturework.
5 Conclusion
Wedemonstratethatobjecttrackingandsegmentationcanemergefromutilizinginformationabout
scenegeometryintheformofdepthsignalsincomplexvideodatawithslot-basedneuralarchitectures.
Weutilizeaseriesofsyntheticmulti-objectvideobenchmarkswithincreasingcomplexitytofinda
simpleyeteffectivesetofchangestoanexistingstate-of-the-artobject-centricvideomodel(SAVi),
allowingustobridgethegapfromsynthetictocomplexreal-worlddrivingvideos.
Our work marks a first step towards building end-to-end trainable systems that learn to perceive
theworldinanobject-centric,decomposedfashionwithoutrelyingondetailedhumansupervision.
Whilemanyopenchallengesremain,thisresultevidencesthatobject-centricdeepneuralnetworks
arenotinherentlylimitedtosimplesyntheticenvironments,andweareexcitedaboutthepotential
forthisclassofmethodstoradicallyreducetheneedforhumansupervisioninbuildingscalable
perceptualsystemsfortherealworld.
6 Acknowledgements
WewouldliketothankBenCaine,AlexBewleyandPeiSunforassistancewithself-drivingdata.
WearegratefultoJieTan,DanielKeysers,DavidFleet,MatthiasMinderer,MehdiSajjadiandMario
Lucˇic´ forgeneraladviceandfeedback.
10

References
[1] ZhipengBao,PavelTokmakov,AllanJabri,Yu-XiongWang,AdrienGaidon,andMartialHebert. Discov-
eringobjectsthatcanmove. InIEEEConferenceonComputerVisionandPatternRecognition(CVPR),
2022.
[2] DanielMBear,ChaofeiFan,DamianMrowca,YunzhuLi,SethAlter,AranNayebi,JeremySchwartz,
LiFei-Fei,JiajunWu,JoshuaBTenenbaum,etal. Learningphysicalgraphrepresentationsfromvisual
scenes. InAdvancesinNeuralInformationProcessingSystems,2020.
[3] JamesBradbury,RoyFrostig,PeterHawkins,MatthewJamesJohnson,ChrisLeary,DougalMaclaurin,
GeorgeNecula,AdamPaszke,JakeVanderPlas,SkyeWanderman-Milne,andQiaoZhang. JAX:compos-
abletransformationsofPython+NumPyprograms,2018. URLhttp://github.com/google/jax.
[4] ThomasBroxandJitendraMalik. Objectsegmentationbylongtermanalysisofpointtrajectories. In
EuropeanConferenceonComputerVision,pages282–295.Springer,2010.
[5] ChristopherPBurgess,LoicMatthey,NicholasWatters,RishabhKabra,IrinaHiggins,MattBotvinick,
andAlexanderLerchner. MONet:Unsupervisedscenedecompositionandrepresentation. arXivpreprint
arXiv:1901.11390,2019.
[6] SergiCaelles,JordiPont-Tuset,FedericoPerazzi,AlbertoMontes,Kevis-KokitsiManinis,andLucVan
Gool. The 2019 DAVIS challenge on vos: Unsupervised multi-object segmentation. arXiv preprint
arXiv:1905.00737,2019.
[7] NicolasCarion,FranciscoMassa,GabrielSynnaeve,NicolasUsunier,AlexanderKirillov,andSergey
Zagoruyko. End-to-endobjectdetectionwithtransformers. InEuropeanConferenceonComputerVision,
2020.
[8] NicolasCarion,FranciscoMassa,GabrielSynnaeve,NicolasUsunier,AlexanderKirillov,andSergey
Zagoruyko. End-to-endobjectdetectionwithtransformers. InEuropeanConferenceonComputerVision,
pages213–229.Springer,2020.
[9] AchalDave,TarashaKhurana,PavelTokmakov,CordeliaSchmid,andDevaRamanan. Tao:Alarge-scale
benchmarkfortrackinganyobject. InEuropeanconferenceoncomputervision,pages436–454.Springer,
2020.
[10] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas
Unterthiner,MostafaDehghani,MatthiasMinderer,GeorgHeigold,SylvainGelly,etal. Animageis
worth16x16words:Transformersforimagerecognitionatscale. InInternationalConferenceonLearning
Representations,2021.
[11] J.Driver,P.McLeod,andZ.Dienes. Motioncoherenceandconjunctionsearch:Implicationsforguided
searchtheory. PerceptionandPsychophysics,51:79–85,1992.
[12] J.T.EnnsandR.A.Rensink. Influenceofscene-basedpropertiesonvisualsearch. Science,247:721–723,
1990.
[13] J.T.EnnsandR.A.Rensink. Sensitivitytothree-dimensionalorientationinvisualsearch. Psychological
Science,1(5):323–326,1990.
[14] HuanFu,MingmingGong,ChaohuiWang,KayhanBatmanghelich,andDachengTao. Deepordinal
regressionnetworkformonoculardepthestimation. InCVPR,June2018.
[15] RohitGirdharandDevaRamanan. Cater:Adiagnosticdatasetforcompositionalactionsandtemporal
reasoning. arXivpreprintarXiv:1910.04744,2019.
[16] Google Research. Google scanned objects, 2020. URL https://app.ignitionrobotics.org/
GoogleResearch/fuel/collections/Google%20Scanned%20Objects.
[17] Klaus Greff, Sjoerd van Steenkiste, and Jürgen Schmidhuber. Neural expectation maximization. In
AdvancesinNeuralInformationProcessingSystems,pages6691–6701,2017.
[18] KlausGreff,RaphaëlLopezKaufman,RishabhKabra,NickWatters,ChristopherBurgess,DanielZoran,
LoicMatthey,MatthewBotvinick,andAlexanderLerchner. Multi-objectrepresentationlearningwith
iterativevariationalinference. InInternationalConferenceonMachineLearning,pages2424–2433,2019.
[19] KlausGreff,SjoerdvanSteenkiste,andJürgenSchmidhuber. Onthebindingprobleminartificialneural
networks. arXivpreprintarXiv:2012.05208,2020.
11

[20] KlausGreff,FrancoisBelletti,LucasBeyer,CarlDoersch,YilunDu,DanielDuckworth,DavidJFleet,Dan
Gnanapragasam,FlorianGolemo,CharlesHerrmann,ThomasKipf,AbhijitKundu,DmitryLagun,Issam
Laradji,Hsueh-Ti(Derek)Liu,HenningMeyer,YishuMiao,DerekNowrouzezahrai,CengizOztireli,
Etienne Pot, Noha Radwan, Daniel Rebain, Sara Sabour, Mehdi S. M. Sajjadi, Matan Sela, Vincent
Sitzmann,AustinStone,DeqingSun,SuhaniVora,ZiyuWang,TianhaoWu,KwangMooYi,Fangcheng
Zhong,andAndreaTagliasacchi. Kubric:ascalabledatasetgenerator. InIEEEConferenceonComputer
VisionandPatternRecognition(CVPR),2022.
[21] AdamWHarley,YimingZuo,JingWen,AyushMangal,ShubhankarPotdar,RitwickChaudhry,and
KaterinaFragkiadaki.Track,check,repeat:AnEMapproachtounsupervisedtracking.InIEEEConference
onComputerVisionandPatternRecognition(CVPR),2021.
[22] KaimingHe,XiangyuZhang,ShaoqingRen,andJianSun. Deepresiduallearningforimagerecognition.
InIEEEConferenceonComputerVisionandPatternRecognition(CVPR),2016.
[23] JonathanHeek,AnselmLevskaya,AvitalOliver,MarvinRitter,BertrandRondepierre,AndreasSteiner,
andMarcvanZee. Flax:AneuralnetworklibraryandecosystemforJAX,2020. URLhttp://github.
com/google/flax.
[24] OlivierJHénaff,SkandaKoppula,EvanShelhamer,DanielZoran,AndrewJaegle,AndrewZisserman,
JoãoCarreira,andReljaArandjelovic´. Objectdiscoveryandrepresentationnetworks. arXivpreprint
arXiv:2203.08777,2022.
[25] ToddSHorowitz,JeremyMWolfe,JenniferSDiMase,andSarahBKlieger. Visualsearchfortypeof
motionisbasedonsimplemotionprimitives. Perception,36(11):1624–1634,2007.
[26] PeterJ.Huber. Robustestimationofalocationparameter. TheAnnalsofMathematicalStatistics,35(1):73
–101,1964.
[27] LawrenceHubertandPhippsArabie. Comparingpartitions. JournalofClassification,2(1):193–218,1985.
[28] AllanJabri,AndrewOwens,andAlexeiAEfros.Space-timecorrespondenceasacontrastiverandomwalk.
InAdvancesinNeuralInformationProcessingSystems,2020.
[29] JindongJiang, SepehrJanghorbani, GerarddeMelo, andSungjinAhn. SCALOR:Generativeworld
modelswithscalableobjectrepresentations. InInternationalConferenceonLearningRepresentations,
2020.
[30] Rishabh Kabra, Daniel Zoran, Loic Matthey Goker Erdogan, Antonia Creswell, Matthew Botvinick,
AlexanderLerchner,andChristopherP.Burgess. SIMONe:View-invariant,temporally-abstractedobject
representationsviaunsupervisedvideodecomposition. InAdvancesinNeuralInformationProcessing
Systems,2021.
[31] DanielKahneman,AnneTreisman,andBrianJGibbs. Thereviewingofobjectfiles: Object-specific
integrationofinformation. Cognitivepsychology,24(2):175–219,1992.
[32] Aishwarya Kamath, Mannat Singh, Yann LeCun, Ishan Misra, Gabriel Synnaeve, and Nicolas Car-
ion. MDETR – Modulated Detection for End-to-End Multi-Modal Understanding. arXiv preprint
arXiv:2104.12763,2021.
[33] LaurynasKarazija,IroLaina,andChristianRupprecht. ClevrTex:Atexture-richbenchmarkforunsuper-
visedmulti-objectsegmentation. InNeurIPSTrackonDatasetsandBenchmarks,2021.
[34] WillKay,JoaoCarreira,KarenSimonyan,BrianZhang,ChloeHillier,SudheendraVijayanarasimhan,
FabioViola,TimGreen,TrevorBack,PaulNatsev,etal. Thekineticshumanactionvideodataset. arXiv
preprintarXiv:1705.06950,2017.
[35] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International
ConferenceonLearningRepresentations,2015.
[36] ThomasKipf,ElisevanderPol,andMaxWelling. Contrastivelearningofstructuredworldmodels. In
InternationalConferenceonLearningRepresentations,2020.
[37] ThomasKipf,GamaleldinFElsayed,AravindhMahendran,AustinStone,SaraSabour,GeorgHeigold,
RicoJonschkowski,AlexeyDosovitskiy,andKlausGreff. Conditionalobject-centriclearningfromvideo.
InInternationalConferenceonLearningRepresentations,2022.
[38] AlexanderKolesnikov,LucasBeyer,XiaohuaZhai,JoanPuigcerver,JessicaYung,SylvainGelly,andNeil
Houlsby. Bigtransfer(BiT):Generalvisualrepresentationlearning. InEuropeanConferenceonComputer
Vision,pages491–507.Springer,2020.
12

[39] Adam Kosiorek, Hyunjik Kim, Yee Whye Teh, and Ingmar Posner. Sequential attend, infer, repeat:
Generativemodellingofmovingobjects. InAdvancesinNeuralInformationProcessingSystems,pages
8606–8616,2018.
[40] HamidLaga,LaurentValentinJospin,FaridBoussaid,andMohammedBennamoun. Asurveyondeep
learningtechniquesforstereo-baseddepthestimation.IEEETransactionsonPatternAnalysisandMachine
Intelligence,2020.
[41] BrendenMLake,TomerDUllman,JoshuaBTenenbaum,andSamuelJGershman. Buildingmachines
thatlearnandthinklikepeople. Behavioralandbrainsciences,40,2017.
[42] FrancescoLocatello,DirkWeissenborn,ThomasUnterthiner,AravindhMahendran,GeorgHeigold,Jakob
Uszkoreit,AlexeyDosovitskiy,andThomasKipf. Object-centriclearningwithslotattention. InAdvances
inNeuralInformationProcessingSystems,2020.
[43] IlyaLoshchilovandFrankHutter. SGDR:Stochasticgradientdescentwithwarmrestarts. InInternational
ConferenceonLearningRepresentations,2017.
[44] SindyLöwe,KlausGreff,RicoJonschkowski,AlexeyDosovitskiy,andThomasKipf. Learningobject-
centricvideomodelsbycontrastingsets. arXivpreprintarXiv:2011.10287,2020.
[45] TimMeinhardt,AlexanderKirillov,LauraLeal-Taixe,andChristophFeichtenhofer. TrackFormer:Multi-
ObjectTrackingwithTransformers. arXivpreprintarXiv:2101.02702,2021.
[46] YueMing,XuyangMeng,ChunxiaoFan,andHuiYu. Deeplearningformonoculardepthestimation:A
review. Neurocomputing,2021.
[47] K.NakayamaandG.H.Silverman. Serialandparallelprocessingofvisualfeatureconjunctions. Nature,
320:264–265,1986.
[48] PeterOchsandThomasBrox.Objectsegmentationinvideo:ahierarchicalvariationalapproachforturning
pointtrajectoriesintodenseregions. InIEEEConferenceonComputerVisionandPatternRecognition
(CVPR),2011.
[49] FedericoPerazzi,JordiPont-Tuset,BrianMcWilliams, LucVanGool, MarkusGross, andAlexander
Sorkine-Hornung. Abenchmarkdatasetandevaluationmethodologyforvideoobjectsegmentation. In
ProceedingsoftheIEEEconferenceoncomputervisionandpatternrecognition,pages724–732,2016.
[50] JordiPont-Tuset,FedericoPerazzi,SergiCaelles,PabloArbeláez,AlexanderSorkine-Hornung,andLuc
VanGool. The2017DAVISchallengeonvideoobjectsegmentation. arXivpreprintarXiv:1704.00675,
2017.
[51] WilliamMRand. Objectivecriteriafortheevaluationofclusteringmethods. JournaloftheAmerican
StatisticalAssociation,66(336):846–850,1971.
[52] RenéRanftl,AlexeyBochkovskiy,andVladlenKoltun. Visiontransformersfordenseprediction. InICCV,
October2021.
[53] BernhardSchölkopf,FrancescoLocatello,StefanBauer,NanRosemaryKe,NalKalchbrenner,Anirudh
Goyal,andYoshuaBengio. Towardcausalrepresentationlearning. ProceedingsoftheIEEE,109(5):
612–634,2021.
[54] E.S.Spelke. Principlesofobjectperception. CognitiveScience,14:29–56,1990.
[55] ElizabethSSpelkeandKatherineDKinzler. Coreknowledge. Developmentalscience,10(1):89–96,2007.
[56] PeiSun,HenrikKretzschmar,XerxesDotiwalla,AurelienChouard,VijaysaiPatnaik,PaulTsui,James
Guo,YinZhou,YuningChai,BenjaminCaine,etal. Scalabilityinperceptionforautonomousdriving:
Waymoopendataset. InIEEEConferenceonComputerVisionandPatternRecognition(CVPR),2020.
[57] ChristianSzegedy,WeiLiu,YangqingJia,PierreSermanet,ScottReed,DragomirAnguelov,Dumitru
Erhan,VincentVanhoucke,andAndrewRabinovich. Goingdeeperwithconvolutions. InCVPR,pages
1–9,2015.
[58] HaoTian,YuntaoChen,JifengDai,ZhaoxiangZhang,andXizhouZhu. Unsupervisedobjectdetection
withlidarclues. InIEEEConferenceonComputerVisionandPatternRecognition(CVPR),2021.
[59] DmitryUlyanov,AndreaVedaldi,andVictorLempitsky. Deepimageprior. InProceedingsoftheIEEE
conferenceoncomputervisionandpatternrecognition,pages9446–9454,2018.
13

[60] SjoerdvanSteenkiste,MichaelChang,KlausGreff,andJürgenSchmidhuber.Relationalneuralexpectation
maximization:Unsuperviseddiscoveryofobjectsandtheirinteractions. InInternationalConferenceon
LearningRepresentations,2018.
[61] AshishVaswani,NoamShazeer,NikiParmar,JakobUszkoreit,LlionJones,AidanNGomez,Łukasz
Kaiser,andIlliaPolosukhin. Attentionisallyouneed. InAdvancesinNeuralInformationProcessing
Systems,pages5998–6008,2017.
[62] RishiVeerapaneni,JohnDCo-Reyes,MichaelChang,MichaelJanner,ChelseaFinn,JiajunWu,Joshua
Tenenbaum, andSergeyLevine. Entityabstractioninvisualmodel-basedreinforcementlearning. In
ConferenceonRobotLearning,pages1439–1456,2020.
[63] AntoninVobecky,DavidHurych,OrianeSiméoni,SpyrosGidaris,AndreiBursuc,PatrickPérez,andJosef
Sivic. Drive&segment:Unsupervisedsemanticsegmentationofurbanscenesviacross-modaldistillation.
arXivpreprintarXiv:2203.11160,2022.
[64] LijunWang,JianmingZhang,OliverWang,ZheLin,andHuchuanLu. SDC-depth:Semanticdivide-and-
conquernetworkformonoculardepthestimation. InCVPR,2020.
[65] YuxinWuandKaimingHe. Groupnormalization. InEuropeanConferenceonComputerVision(ECCV),
pages3–19,2018.
[66] RuibinXiong,YunchangYang,DiHe,KaiZheng,ShuxinZheng,ChenXing,HuishuaiZhang,Yanyan
Lan,LiweiWang,andTieyanLiu. Onlayernormalizationinthetransformerarchitecture. InInternational
ConferenceonMachineLearning,2020.
[67] JiaruiXu,ShaliniDeMello,SifeiLiu,WonminByeon,ThomasBreuel,JanKautz,andXiaolongWang.
Groupvit:Semanticsegmentationemergesfromtextsupervision. arXivpreprintarXiv:2202.11094,2022.
[68] CharigYang,HalaLamdouar,ErikaLu,AndrewZisserman,andWeidiXie. Self-supervisedvideoobject
segmentationbymotiongrouping. InProceedingsoftheIEEEInternationalConferenceonComputer
Vision,2021.
[69] YiZhou,HuiZhang,HanaLee,ShuyangSun,PingjunLi,YangguangZhu,ByungInYoo,XiaojuanQi,
andJae-JoonHan. Slot-vps:Object-centricrepresentationlearningforvideopanopticsegmentation. In
ProceedingsoftheIEEE/CVFConferenceonComputerVisionandPatternRecognition,pages3093–3103,
2022.
[70] DanielZoran,RishabhKabra,AlexanderLerchner,andDaniloJRezende. Parts:Unsupervisedsegmen-
tationwithslots,attentionandindependencemaximization. InProceedingsoftheIEEEInternational
ConferenceonComputerVision,pages10439–10447,2021.
14

A Societalimpact
Our work focuses on extending object-centric representation learning to real-world videos. This
classofmethodshasthepotentialforenablingsystemstomorereliablysolvedown-streamtasks
requiringobjectrepresentations,suchasrelationalreasoningoverentitiesinvideos,andproviding
better interpretation of model decisions. Some applications that may benefit from this approach
includeperceptioninautonomousvehicles,roboticsandclassiccomputervisionproblemssuchas
objectdetection. Whilethemethoddemonstratedinthispaperisstillfarfromastatewhereitcould
bedirectlyemployedincomputervisionapplications,wewouldliketoraiseawarenessthat—aswith
mostmethodsdevelopedforcomputervision—advancesinthisfieldmightalsoaidthedevelopment
applicationswithpotentialnegativesocietalimpactsuchassurveillance.
B Additionalresults
Qualitativeresults InFigure7,weshowtheeffectofourmaskthresholdingheuristicappliedfor
ourunsupervisedmodelvisualizations. Theintentionforthissimpleheuristicistoaidinterpretability
of the discovered object segmentation masks. We further show qualitative results for emergent
trackingonlongsequencesintheunconditionalsetting(i.e.withoutboundingboxconditioning)in
Figure8. Comparedtotheconditionalsetting(shownforreference),trackingislessconsistentand
slotsexplainnotonlycars,butalsoenvironmentalobjectsorpartofthebackground.
Input frame SIMONe SIMONe + depth SAVi++ (unconditional) SAVi++ (conditional)
Figure 7: Comparison of qualitative result visualization without (top) and with (bottom) mask
thresholding. Weapplyasimplethresholdingheuristicofdroppinganymasksthat(onaverageacross
frames)occupymorethan1300pxperframe,whichaidsinterpretability. Thresholdingdoesnothave
aneffectontheconditionalmodel.
t = 0 t = 5 t = 10 t = 15 t = 20 t = 25
lanoitidnocnU
lanoitidnoC
Figure8: QualitativeresultsonlongWaymoOpenvalidationsetvideos(forSAVi++modelstrained
on 6 frames). Top: Results for an unconditional SAVi++ model, trained and evaluated without
boundingboxconditioninginthefirstframe. Bottom: Conditionalsettingshownforreference.
15

Quantitativeresults WeadditionallycompareSAViandSAVi++onthesyntheticMOVi-Aand
MOVi-Bdatasets[20]. InMOVi-A,scenesconsistofagrayfloor,fourlightsources,afixedstatic
cameraandbetween3and10simplegeometricobjectsthatvaryintermsoftheirshape(cube,sphere,
cylinder), material (rubber, metal), size (small, large), and color (blue, brown, cyan, gray, green,
purple,red,yellow). MOVi-Bisastraightforwardextension,whichaddsadditionalvariationtothe
objectshape,size,andcolor;backgroundcolor;andstaticcamerapositions.
In Table 3 it can be seen how SAVi++ performs similar or worse in terms of mIoU and FG-ARI
on these datasets. We mainly attribute this to our strategy for scaling in SAVi++, which appears
susceptibletooverfittingonthesemuchsimplerdomains. Additionally,thebenefitofusingdepth
informationislimited,sinceallobjectsareinmotionforthesedatasets. Thecontrastbetweenthe
resultspresentedinthemainpaperforMOVi-Cto-EandTable3(MOVi-Aand-B)emphasizes
theimportanceofconsideringbenchmarksthataremorerepresentativeoftherealworldformodel
development.
Table3: MOViresultsintermsofmeanscore±standarderror(5seeds)fromevaluatingSAVi++and
SAVimodelsonvalidationsetvideosequencesofincreasedlength(24frames).
|          |          | mIoU↑(%)  |          | FG-ARI↑(%) |
| -------- | -------- | --------- | -------- | ---------- |
| Model    | MOVi-A   | MOVi-B    | MOVi-A   | MOVi-B     |
| SAVi[37] | 82.3±0.3 | 44.5±9.3  | 96.8±0.4 | 73.9±10.7  |
| SAVi++   | 76.1±0.9 | 25.8±11.3 | 98.2±0.2 | 48.3±15.7  |
Table4: BreakdownofSAVi++resultsfromTable2intermsofthreeclassesofobjects: car,person,
andcyclist.
| Metric    |         | Car      | Person   | Cyclist  |
| --------- | ------- | -------- | -------- | -------- |
| Num.      | objects | 15350    | 2102     | 275      |
| CoM↓      |         | 4.2±0.2  | 6.4±0.1  | 1.9±0.1  |
| B.mIoU↑   |         | 52.5±0.8 | 27.0±0.5 | 42.2±2.1 |
| B.Recall↑ |         | 96.7±0.7 | 95.1±1.1 | 99.3±0.3 |
WereportpercategoryresultsforSAVi++ontheWaymoOpendatasetinTable4. Wefindthat,as
expected,thisdatasetisdominatedbycarsandperformanceisverygoodinthiscategory. SAVi++
alsoperformsverywelloncyclists,whichareveryrareinthisdataset,andindicatesthatSAVi++has
notoverfittoblobbycarlikeobjects.
WefurtherreportresultsforSAVi++undertheinfluenceofnoiseinthesparsedepthtargetsonWaymo
OpeninTable5. Ourresultsindicatethatemergenttrackingperformanceislargelyunaffectedby
noisescales(standarddeviations)ofuptoσ =40cm.
Supplementaryvideos Weprovideseveralvideoresultsinthesupplementarymaterialforboth
SAVi++ (conditional, uncondtional, and high-resolution model variants) and the SIMONe [30]
baseline(bothinitsoriginalformandinouradapteddepth-predictionvariant).
Table5: WaymoOpenresults(mean±standarderrorin%, 3seeds)fromevaluatingmodelson
sequencesof10frames. SAVi++wastrainedwithnoisydepthtargetswithstandarddeviation(σ)
specifiedbelow.
(%)
| Model    |        | CoM↓    | B.mIoU↑  | B.Recall↑ |
| -------- | ------ | ------- | -------- | --------- |
| SAVi++   |        | 4.4±0.2 | 49.7±0.7 | 96.5±0.7  |
| SAVi++(σ | =10cm) | 4.3±0.2 | 50.1±0.3 | 96.8±0.3  |
| SAVi++(σ | =20cm) | 4.3±0.2 | 49.7±0.3 | 97.4±0.5  |
| SAVi++(σ | =40cm) | 4.2±0.0 | 50.1±0.3 | 96.9±0.5  |
16

C Trainingsetup
Wetrainourmodelsfor500ksteps(300kstepsfortheablationstudy)onTensorProcessingUnit
(TPU)acceleratorswithabatchsizeof64usingAdam[35]. Welinearlyincreasethelearningratefor
2500stepsto0.0002(startingfrom0)andthendecaythelearningratewithaCosineschedule[43]
backto0fortherestofthetrainingsteps. Weclipthegradientstoaglobalnormvalueof0.05to
stabilizetraining. Tocreatevideoexamplesfortraining,wespliteachvideointosub-sequencesof6
frameseach.Weuseatotalof24slotsonMOViand11slotsonWaymoOpenforSAVi++models.We
use1iterationperframefortheSlotAttention[42]module(asinpriorwork),unlessstatedotherwise.
FollowingtheconditionalsetupinKipfetal.[37],theinitialstateoftheslotsisobtainedbyencoding
boundingboxescorrespondingtotheobjectsinthefirstframe;thusprovidingthemodelwithrough
cuesofwhichobjectstobindtoinitially. Fortheunconditionalexperiments,weinitializeslotsusing
anequalamountoflearnableparametervectors. Inexperimentsthatuseopticalflow,weconvertthe
2DflowsignaltothreeRGBchannelsfollowingpriorwork[68]. Asdescribedabove,weapplya
log-transformtothe(sparse)depthsignal(incrementedby1toavoidunderflow). Wetrainourmodel
tominimizethesquarederror(L2loss)betweenthepredictedandground-truthtargetsinpixelspace.
WeimplementSAVi++inJAX[3]usingtheFlax[23]neuralnetworklibrary. TrainingSAVi++ona
singleMOVidataseton8TPUv4chipswith32GiBmemoryeachtakesapproximatelytwodaysfor
500ktrainingsteps.
D Modeldetails
D.1 SAVi++
OurarchitecturebuildingblocksaresimilartothatoftheSAVimodelfromKipfetal.[37]with
alltheparameterssharedacrosstimesteps. SAVi++usesexactlythesameparametersfortheslot
initializeranddecoderastheSAVimodel(exceptforSAVi++HRwhereweuseanadditional5×5
ConvTransposelayerwithstride2and64channelstoaccountforthehigherresolutionframesize).
BelowwelistthedetailsandhyperparametersofallthemodulesofSAVi++:
Encoder We used a ResNet-34 [22] backbone with modified root convolutional layer that has
1×1stride(exceptforSAVi++HRthatusesarootstrideof2×2). Foralllayers,wereplacedthe
batchnormalizationoperationbygroupnormalization[65]. Weusedalinearpositionalencoding
identicaltothatusedinSlotAttention[42]withhorizontalandverticalcoordinatesnormalizedto
[−1,1]range. ThesecoordinateswerethenprojectedtothesamesizeoftheResNetfeaturemaps
usingalearnablelinearlayer. Finally,theResNetfeaturesandpositionalencodingarecombinedby
anadditionoperation. Followingthebackbone,theframefeaturesareprojectedto64embedding
dimensionsbyalinearlayerfollowedbyReLUactivationandthenfedtoatransformernetwork
with4transformerblocks,exceptforSAVi++unconditionalWaymoOpenmodelswherewefound
thatResNetfeaturesaloneproducedclearersegmentations. ThisislikelyduetothestrongerCNN
imagepriorbeingbeneficialwithlesssupervision[59]. Eachtransformerblockusesamulti-head
dot-productattentionfrom[61]withpre-normalization[66]and4attentionheads. Foreachattention
head,thequery/key/valueembeddingsizewassetto16. Theoutputofeachblockisthenprocessed
byaresidualfeed-forwardblockwithpre-normalization,usinganMLPwithasinglehiddenlayerof
1024hiddenunitsandReLUactivationfunction.
Corrector/Predictor Similar to SAVi [37], we use 1 iteration for the Slot Attention corrector
module. Weincreasethecorrectorquery/key/valuesizeto256comparedto128embeddingsizeused
inSAVi. Forthepredictor,wesimilarlyincreasedthequery/key/valueprojectionsizeto256,and
theMLPhiddenlayersizeto1024. Wefoundthatthelargerembeddingsizesforthecorrectorand
predictorincreasedSAVi++mIoUbyafewpercentagepoints.
Decoder OurdecoderfollowsthatofSAVi[37]withtwoexceptions: ForWaymoOpen,weusea
largerspatialbroadcastgridof8×12asvideoframeshave128×192resolution. ForSAVi++HR,
weuseframesofahigherresolutionof256×384andaddanadditional5×5ConvTransposelayer
withstride2and64channelstoaccountforthehigherresolutionframesize. Thedecoderotherwise
usesfour5×5ConvTransposelayerswithstride2and64channels,followedbyReLUactivations.
17

Initializer SimilartoSAVi,weconsidertwoinitializersconditionalandunconditionaltosetthe
initial state of the models’ K slots. For the conditional case, the initializer mapped each of K
boundingboxes(representedusing4Dcoordinates)viaatrainableMLPtosettheDdimensional
stateofthecorrespondingslot. Fortheunconditionalcase,ratherthanassociatingaspecificbounding
boxinaspecificvideotoaslot,welearnK D-dimensionalinitialslotstatestobeusedinallvideos.
Dataaugmentation Asdiscussedinthemaintext,anInceptionstyle[57]randomcropisusedfor
dataaugmentation. Cropsareafterwardsresizedtothetargetresolution(128×128forMOViand
128×192forWaymoOpen,unlessotherwisementioned). Weensurethatthecroppedviewcoversat
least20%oftheoriginalframeintheMOVidatasetsand75%inWaymoOpen. Wefoundthatmore
aggressivecroppingwashelpfulforthesyntheticMOVidatasetsandlessaggressivecroppingworked
bestonWaymoOpen. Wetakecaretohandledepthmapsandflowfieldsduringthisoperationas
explainednextforeachofthedatasets. MOVi: Saythecropsizeish×w,thenopticalflowfields
arecropped,resized,andthenrescaledby
(cid:2)128,128(cid:3)
. Densedepthmapsaresimplycroppedand
w h
resized. WaymoOpen: SparseLiDARpointcloudsareprojectedto2Dandretainedastuples(2D
point,lidar-range)throughoutthedataaugmentationpipeline. Theaffinetransformationequivalentto
thecropandresizeoperationiscomputedandappliedtothese2Dpoints. Asaconsequence,several
pointswillfalloutsidethecroppedframe. Thesepointsarediscardedwhenprojectingthemintoa
depthimageafterdataaugmentationiscomplete.
D.2 Baselines
SAVi For the SAVi baseline, we use the best-performing model variant described in Kipf et al.
[37],i.e.SAVitrainedwithaResnet34backbone. DifferentfromSAVi++,thisbaselinedoesnotuse
depthprediction(itonlypredictsopticalflow),doesnotusedataaugmentation,anddoesnotuse
atransformerencoderaftertheconvolutionalbackbone. Wechoosethesamehyperparametersas
describedinSAVi[37].
SIMONe This baseline [30] is a non-autoregressive model for encoding short video clips of
fixedlengthsintoasetoflatentobjectvariables(fixedacrosstime)andaper-framegloballatent
variable. NotethatSIMONecannotbeappliedauto-regressivelyandhastobeappliedtothesame
sequencelengthatbothtrainingandtesttime. SIMONeusesaCNNencoderperframefollowed
by a transformer encoder that is applied across frames to finally obtain object and frame latent
variablesbypoolingtransformertokensacrosstimeandspace,respectively. Themodelistrainedby
reconstructinginputframesusingaformofaspatialbroadcastdecoderandadditionallyusesaKL-
basedregularizeronthelatentvariables. WeuseaJAX[3]reimplementationoftheSIMONemodel
forwhichweverifiedthatitreproducesresultsmentionedinthepaperontheCATER[15]dataset.
WetrainSIMONeonsequencesof6frames(sameasSAVi++)ataresolutionof128. Wesubsample
reconstructiontargetsbyafactorof4(seeKabraetal.[30]fordetails). Otherhyperparametersare
chosenasfollows: reconstructionlossscaleα=0.2,pixellikelihoodscaleσ =0.08,objectlatents
x
KLlossweightβ = 1e−5,andframelatentsKLlossweightβ = 1e−4. Weencodeframes
o f
usinga4-layerCNNwith128channels,(4,4)kernelsizeand(2,2)stride. Eachtransformeruses4
layers,5heads,aqkv-sizeof64perhead,andanMLPhiddenlayersizeof1024. Latentsareofsize
32. ThedecoderusesanMLPwith5hiddenlayersof512units. TotrainSIMONewithsparsedepth
targets,wereplacetheRGBtargetsignalwiththeLiDAR-baseddepthsignalandonlycomputethe
lossforpixelsthathaveadepthsignal.
CRW ContrastiveRandomWalks(CRW)[28]isacycleconsistencybasedself-supervisedlearning
methodforlearninggridstructuredlatentrepresentations. Afterpre-training,asimplelabelpropaga-
tionschemecanbeappliedontheselatentrepresentationstoobtaintrackingbehavior. Thistypically
requiressegmentationlabelsfortheobjectsofinterestinthefirstframe. Themethodtracksthese
objectsandoutputssegmentationmasksforthemoversubsequentframes. Inordertousethismethod
withonlyboundingboxconditioninginthefirstframe,wefloodfillboxesintorectangularmasksand
propagatethoseinstead. Overlapbetweenboxesisresolvedbasedontheboxorder.
We adopted their training and evaluation best practices. We pre-trained stride-8 ResNet back-
bones[22]usingtheirpubliclyavailablecodeandpropagatedlabelsusingtheactivationsoutputby
thesecondlastResNetstage. Forpre-trainingwetunededge-dropout,trainingtemperatureandfor
trackingwetunedevaluationtemperatureindependentlyoneachofthethreeMOVidatasets. We
18

foundthat,despiteourefforts,aResNet34backbonewasnotabletotrainusingthecycleconsistency
loss. WeobtainedmuchbetterresultsusingaResNet18backbone,whichisthemodelwereport
resultsfor. Optimalhyper-parameters(dropout,trainingtemperature,evaluationtemperature)wereas
follows: MOVi-C(0.0,0.001,0.5),MOVi-D(0.05,0.001,0.5),MOVi-E(0.05,0.001,0.5). Other
relevanthyper-parametersare: trainingcliplength(6),frame-skip(1),batchsize(16),learningrate
(0.0001),trainingepochs(125)withalearningratedropafterthe100thepoch.
Boundingboxcopy Wesimplyrepeattheboundingboxesoftheobjectsvisibleintheinitialframe
fortherestofthevideosequence. Toobtainpixel-levelsegmentsforcomputingmetrics,suchas
FG-ARI, we ‘render’ the entire bounding box as a segment in pixel space. Bounding boxes are
rendered in the same order as they were provided in the initial frame, such that later boxes take
precedencewhenmultipleofthemcoverthesamepixellocation.
Learnedboundingboxpropagation Inthisbaseline,weusetheSAVi++initializerandpredictor
(withoutencoder, corrector, ordecoder)tolearnaboundingboxpropagationmodel. Themodel
receives(justasinSAVi++)boundingboxesforallobjectsinthefirstframeofthevideo,whichare
passedtotheinitializertolearninitialslotrepresentations.Afterwards,thepredictorlearnsamapping
ofslotsattimestepttoslotsattimestept+1. Wetrainthemodelbyreadingoutindividualslot
representationsateachtimestepusinganMLPwithasinglehiddenlayerof256unitsthatpredicts
thecornercoordinates(top-leftandbottom-right)oftheboundingboxassociatedwithaslotata
particulartimestep,supervisedusingground-truthboundingboxes. WeusetheHuber[26]loss(L2
lossbetween[−1,1]andL1lossoutsideofthisinterval)totrainthemodel. Ifanobjectisnotvisible
orpresent,itsboundingboxissetto[0,0,0,0]intheground-truthtarget.
K-Meansclusteringofflow/depth Toevaluatehowmuchinformationaboutinstancesegmentation
canbeexctracteddirectlyfromthedepthandopticalflowmodalities,weevaluateak-Meansclustering
baselineonvideosfromMOViandWaymoOpen. Forthatpurposewetreateachpixelofavideoas
adatapoint,eachwith7dimensions: oneforlog-depth(log1+d),threeforopticalflowconverted
toRGB(onlyforMOVi),twoforlinearpositionencoding,andonefortime. Alldimensionsare
normalizedtotherangeof[0,1].ForWaymoOpenwediscardallpointsthatdonothaveanassociated
depthvalue. TomakeitascomparableaspossibletotheconditionalsetupofSAVi++,wesetktothe
ground-truthnumberofobjectsplusoneforthebackground,andinitializeeachcluster-centertothe
averagevalueofpointswithinthefirst-frameboundingboxofeachobject. Thebackgroundclusteris
initializedtotheaveragevalueofallpointsinthefirstframe. K-Meansisthenrununtilconvergence,
andweevaluatetheresultingcluster-assignmentsusingmIoUandFG-ARIscoresforMOVi,and
bycomputingthenormalizeddistanceofthecenterofmassofeachsegmenttothecorresponding
boundingboxforWaymoOpen.
Supervisedbaseline Toestimatehowmuchheadroomthereisintermsoftrackingperformance
givenourmodelarchitecture,wetrainavariantoftheSAVi++modelwherewereplacethedepth
decoder with a bounding-box prediction head. Instead of self-supervised training using depth
prediction,thismodelistrainedtodirectlypredictobjectboundingboxesateverytimestepgiven
theslotsofthemodel. ThisissimilartoaTrackFormer[45]model,butweinsteadusetheSAVi++
architectureandwetraininaconditionalsetting(i.e.initialfirst-frameboundingboxesareprovidedas
slotinitialization),whichmeanswecantrainthemodelwithoutusinganyformofmatching. Similar
tothelearnedboundingboxpropagationbaseline,wetrainthemodelbyreadingoutindividualslot
representationsateachtimestepusinganMLPwithasinglehiddenlayerof256unitsthatpredicts
thecornercoordinates(top-leftandbottom-right)oftheboundingboxassociatedwithaslotata
particulartimestep,supervisedusingground-truthboundingboxes. WeapplyaHuber[26]loss(L2
lossbetween[−1,1]andL1lossoutsideofthisinterval)fortraining. Ifanobjectisnotvisibleor
present,itsboundingboxissetto[0,0,0,0]intheground-truthtarget.
E Datasets
WeusedthesyntheticMulti-ObjectVideo(MOVi)datasetsintroducedinKubric[20]. TheKubric
datasetgenerationpipelineisavailableunderanApache2.0license. Forreal-worldexperiments,we
usedtheWaymoOpendataset[56]. TheWaymoOpendatasetislicensedundertheWaymoDataset
LicenseAgreementforNon-CommercialUse(August2019): https://waymo.com/open/terms.
19

Datasetdetailsaresummarizedinthefollowing:
• MOVi-C:usesapproximately380high-resolutionHDRphotosasbackgroundsandthreetoten
dynamic objects obtained from a set of 1028 3D-scanned everyday objects [16], representing
varioushouseholdobjects. Thecamerainthisdatasethasarandompose,yetthecameraposeis
staticacrossthevideosequence. Eachvideoissampledat12framespersecond(fps). Wetrained
ourmodelsonrandomlysampledsequencesof6framesandevaluateonsequencesof24frames.
Wetrainedourmodelson9.75ktrainingsetvideos,andevaluatedmodelson250evaluationset
videos. Modeltuningwasperformedonaseparatelygeneratedsetof250videos.
• MOVi-D:hasasimilarcamerasettingasMOVi-C,butaddsmoreobjects,themajorityofwhich
areinitializedtobestatic. Thisdatasetincludesonetothreedynamicobjectsand10to20static
objectsineachvideosequence. Weusedthesameframerateandtrain/evaluationsequencelengths
asforMOVi-C.Wetrainedourmodelson9.75ktrainingsetvideos,andevaluatedmodelson250
evaluationsetvideos. Modeltuningwasperformedonaseparatelygeneratedsetof250videos.
• MOVi-E:addsadditionalcomplexitycomparedtoMOVi-Dbyintroducingrandomlinearcamera
movement throughout the video sequence. We used the same frame rate and train/evaluation
sequencelengthsasforMOVi-C.Wetrainedourmodelson9.75ktrainingsetvideos,andevaluated
modelson250evaluationsetvideos. Modeltuningwasperformedonaseparatelygeneratedsetof
250videos.
• WaymoOpen: containshigh-resolutionvideosequenceswithframesizeof1280×1920. We
solely use videos recorded from the front camera of the car. We down-sampled the videos to
128×192(or256×384forSAVi++HR).Thedatasetconsistsof798trainand202validation
scenesof20svideoeach,sampledat10fps.Thedatasetincludesalso2Dboundingboxannotations,
whichweusedfortheconditionalexperimentsandtocomputetheB.mIoUevlauationmetrics.
TheWaymoOpendatasetfurtherincludesLiDARdatacollectedfromfiveLiDARs;onemid-range
LiDARsplacedontopofthecarandfourshort-rangeLiDARsplacedfront,left,right,andrear.
TheLiDARdataisusedtocomputesparsedepthtargetsasdiscussedintheMethodssection. To
slightlysimplifythetaskaswetrainonlower-resolutionframes,wediscardanyboundingbox
labelsinWaymoOpenwhichcoveranareaof0.5%orlessofthefirstsampledvideoframe,both
duringtrainingandtesting.
F Metrics
Inthefollowing,wegiveadetailedoverviewofthemetricsusedforeachofthedatasets.
F.1 MOVi
OntheMOVidatasets[20]wehaveaccesstoground-truthpixel-levelsegmentations,whichletsus
directlymeasurethequalityofthelearnedsegmentationsusingthesamesegmentationmetricsas
inpriorwork. NotethatSAViandSAVi++aretrainedinaconditionalsettingwhereweinitialize
slotsusingground-truthboundingboxinformationinthefirstframe. Becauseofthis,wewillonly
measuremetricsfromthesecondframeonward.
ForegroundAdjustedRandIndex(FG-ARI) Apermutation-invariantclusteringsimilaritymet-
ric frequently used for evaluating scene decomposition quality [27, 51]. It compares discovered
segmentationmaskswithground-truthmaskswhileignoringanypixelsthatbelongtothebackground.
Itissensitivetotemporalconsistencyofmasks,butinsensitivetotheirordering.
MeanIntersectionoverUnion(mIoU) Astandardsegmentationmetricformeasuringthequality
ofpredictedsegments. Ourimplementationisidenticaltothesemi-supervisedDAVISchallenge
Jaccard-Meanmetricforvideo[6,50]. Wenotethatthisimplementationissensitivetothecorrect
orderingofmasks,i.e.italsomeasureswhethermodelsusedtheconditioningsignal(here,first-frame
boundingboxes)correctly.
ModelselectiononMOViwasdonemainlyusingmIoU.Ontheonehandtoavoidlearnedsegments
bleedingintothebackgroundoverlymuch,andontheotherhandtoensurethattheboundingbox
initializationwasproperlyutilized.
20

F.2 WaymoOpen
OntheWaymoOpendatasetweonlyhaveground-truthboundingboxesavailable,whichnecessitates
analternativesetofmetricsformeasuringquantitativeperformance. Similartobefore,becauseof
conditioning,wewillonlymeasuremetricsfromthesecondframeonward.
Center-of-Mass(CoM)distance ThistrackingmetricmeasurestheaverageEuclideandistance
between the centroid of the predicted segmentation masks and the centers of the ground-truth
boundingboxes. Theformerareobtainedbycomputingthegeometricmeanofthe2Dcoordinates
associatedwiththepixelsbelongingtoasegment,whereweexcludepixelsthatdonothaveavalid
LiDARpointassociatedwiththem(thisissimilartohowwecomputethelossduringtraining). To
allowforcomparableCoMdistanceacrossmultipleresolutions,wereportthedistancenormalizedby
themaximumachievabledistanceinthevideoframe(lengthofthediagonal). Objectsthatarefully
occludedinaframe(orhavedisappeared)areexcludedfromthecomputation. Intheconditionalcase
(i.e.withboundingboxinformationprovidedtothemodelinthefirstframe)weusetheorderofthe
providedboundingboxestocomputethemetricbetweeneachslotandground-truthboundingbox.
Intheunconditionalcase,weuseHungarianmatchingtoassociateentireboundingboxtrackswith
decodedobjectmasksandweassignapenaltyof1(maximumCoMdistance)forallemptysegments.
BoundingBoxRecall(B.Recall) Thismetricmeasuresthefractionofcaseswhereanysortof
segmentispredictedwhenavalidground-truthboxexists. ItservesacomplementtoCoMdistance
when no matching is used and empty segments are not considered. In the case of unconditional
evaluationusingHungarianmatching,weincorporateamatchingpenaltyofthemaximumpossible
CoMdistanceforemptysegmentsandthusdonotseparatelyreportsegmentrecall.
Bounding Box mIoU (B. mIoU) This metric is the bounding box analog of the segmentation
mIoUdiscussedabove. Givencorrespondingpredictedandground-truthboundingboxtracks,their
per-frame intersection-over-union is computed and averaged over time exactly as in the average
IoUmetricoftheTAObenchmark[9]. Predictedboundingboxtracksareobtainedusingaper-slot
readoutMLP,withonehiddenlayerof256units. ThisisjointlytrainedwiththeSAVi++model
by minimizing the Huber loss [26] between predictions and [0,1] normalized ground-truth box
coordinates. A stop-gradient is used to prevent these loss gradients from propagating back into
SAVi++. Objects that are fully occluded across the entire video sequence are excluded from the
computation.
We initially conducted model selection on Waymo Open using a combination of B. mIoU and a
heuristicmetrictomeasurewhatfractionofthepixelsbelongingtoapredictedsegmentareinsidethe
associatedground-truthboundingbox. Duringthefinalstagesofdevelopment,weprimarilyfocused
ontheB.mIoUmetricsinceitisanalogoustomIoUonMOVi.
21
