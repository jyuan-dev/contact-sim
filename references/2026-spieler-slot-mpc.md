Slot-MPC: Goal-Conditioned Model Predictive
Control with Object-Centric Representations
JonathanSpieler∗ AngelVillar-Corrales SvenBehnke
AutonomousIntelligentSystems,ComputerScienceInstituteVI-IntelligentSystemsandRobotics,
CenterforRoboticsandtheLamarrInstituteforMachineLearningandArtificialIntelligence,
UniversityofBonn,Germany
Abstract
Predictiveworldmodelsenableagentstomodelscenedynamicsandreasonabout
theconsequencesoftheiractions. Inspiredbyhumanperception,object-centric
worldmodelscapturescenedynamicsusingobject-levelrepresentations,whichcan
beusedfordownstreamapplicationssuchasactionplanning.However,mostobject-
centricworldmodelsandreinforcementlearning(RL)approacheslearnreactive
policiesthatarefixedatinferencetime,limitinggeneralizationtonovelsituations.
WeproposeSlot-MPC,anobject-centricworldmodelingframeworkthatenables
planningthroughModelPredictiveControl(MPC).Slot-MPCleveragesvision
encoderstolearnslot-basedrepresentations,whichencodeindividualobjectsin
thescene,andusesthesestructuredrepresentationstolearnanaction-conditioned
object-centricdynamicsmodel. Atinferencetime,thelearneddynamicsmodel
enablesactionplanningviaMPC,allowingagentstoadapttopreviouslyunseensit-
uations. Sincethelearnedworldmodelisdifferentiable,wecanusegradient-based
MPCtodirectlyoptimizeactions,whichiscomputationallymoreefficientthan
relyingongradient-free,sampling-basedMPCmethods. Experimentsonsimulated
roboticmanipulationtasksshowthatSlot-MPCimprovesbothtaskperformance
and planning efficiency compared to non-object-centric world model baselines.
Intheconsideredofflinesettingwithlimitedstate-actioncoverage,wefindthat
gradient-basedMPCperformsbetterthangradient-free,sampling-basedMPC.Our
resultsdemonstratethatexplicitlystructured,object-centricrepresentationsprovide
astronginductivebiasforcontrollableandgeneralizabledecision-making. Code
andadditionalresultsareavailableathttps://slot-mpc.github.io.
1 Introduction
The ability to predict how the world evolves in response to actions is fundamental to intelligent
behavior. Worldmodels[HaandSchmidhuber,2018,LeCun,2022]aimtoequipartificialagents
withthiscapabilitybylearningpredictivemodelsthatenableforecastingfutureenvironmentstates,
supportinganticipationandplanning. Humans,however,donotperceivetheworldasanunstructured
stream of pixels. Instead, we parse scenes into persistent objects, which move independently,
can interact with each other, or compose into more complex entities [Kahneman et al., 1992].
Inspired by human perception, several works have investigated object-centric and compositional
inductivebiasesforpredictivemodeling,demonstratingdesirablepropertiessuchasgeneralizationto
novelcompositions[Haramatietal.,2024],transferabilitytonoveltasks[Zhangetal.,2022],and
improvementsinsampleefficiency[Mosbachetal.,2025],amongothers.
Basedontheseinsights,thefieldofobject-centriclearninghasrapidlyprogressedinrecentyears,
movingfromlearningobjectrepresentationsonsyntheticimages[Burgessetal.,2019,Locatello
etal.,2020]andvideos[Kipfetal.,2022],towardsmorecomplexreal-worldscenariosanddatasets
∗Correspondencetospieler@ais.uni-bonn.de.
Preprint.
6202
yaM
41
]GL.sc[
1v73941.5062:viXra

(a)cOCVPtraining (b)Slot-MPCinference
Figure 1: Overview of Slot-MPC. (a) The object-centric world model (cOCVP) is trained given
asinglevideoframeX andactionsa,andautoregressivelypredictsfuturevideoframesandslot
1
representations S. (b) Slot-MPC parses an image X into its object representations S and then
1 1
predictsthefutureobjectstatesoverthehorizonH usingcOCVPgivenactionsa,whichareprovided
byMPC.ThegoalimageX isalsoparsedintoitsobjectrepresentationsS andtheslotsattime
G G
stepSˆ andS areusedtooptimizetheactionsusingtheMPCobjectiveJ definedinEq.(9).
T G
[Seitzeretal.,2023,Zadaianchuketal.,2024]. Additionally,thelearnedobjectrepresentationshave
beenusedforworldmodelingandrobotictasks,includingmodel-basedRL[Ferraroetal.,2023,
Mosbachetal.,2025]orimitationlearning[Villar-CorralesandBehnke,2025].
While effective, most object-centric RL policies are fixed at inference time, leading to purely
reactivebehaviorsthatlimitgeneralizationtonovelsituations[Byravanetal.,2022]. Furthermore,
onlineRLapproachesaretypicallynotsample-efficientandrequirelargenumbersofenvironment
interactions,makingthemcostly. Planning-basedcontroloffersacomplementaryalternative. Model
PredictiveControl[CutlerandRamaker,1979,Richaletetal.,1978](MPC)enablesonlineplanning
by optimizing actions using a learned dynamics model at inference time. Recent works such as
TD-MPC2[Hansenetal.,2024]combinepolicynetworkswithMPCtoleveragethebenefitsofboth
paradigms.
Inthiswork,weproposeSlot-MPC,anovelobject-centricframeworkforgoal-conditionedplanning
with MPC. Rather than relying only on a reactive policy, Slot-MPC learns a structured latent
world model that enables online planning directly in an object-centric representation space. At
inference time, as illustrated in Fig. 1b, Slot-MPC parses environment observations into a set of
objectslots,whichrepresentindividualentitiespresentinthescene. Usinganaction-conditioned
object-centricdynamicsmodel,Slot-MPCpredictsfutureobjectstatesfrompastslotsandcandidate
actionsequences,andoptimizesactionsviaMPCbyminimizingthedistancebetweenpredictedand
goalobjectconfigurationsinslotspace. Leveragingthedifferentiabilityofthelearnedworldmodel
forefficientoptimization,candidateactionsequencesaredirectlyoptimizedbygradientdescent. This
formulationenablesplanningdirectlyinastructuredobject-centriclatentrepresentation,ratherthan
relyingonholisticscenefeatures.
Inourexperiments,wedemonstratethatSlot-MPClearnsmeaningfulobjectrepresentations,which
enablemoreefficientplanningcomparedtoholistic,non-object-centricrepresentations. Slot-based
object-centricmodelsreducethelatentspacedimensionalityby99%comparedtopatch-basedap-
proachessuchasDINO-WM[Zhouetal.,2025],whichleadstomoreefficientplanning.Furthermore,
theexplicitdisentanglementofobject-levelentitiesenablesmoredirectandcontrollablereasoning
overscenedynamics,whichisparticularlybeneficialinlow-dataandlow-computeregimes.
Insummary,ourcontributionsareasfollows:
• WeintroduceSlot-MPC,agradient-basedMPCmethodthatusesalatentobject-centricdynamics
modelandslot-basedrepresentationsforgoal-conditionedplanningfrompurelyvisualinputs.
• Weshowthatthelearnedslot-basedrepresentationsenablemoreefficientgradient-basedplanning
comparedtoholisticandpatch-basedlatentrepresentations.
• WedemonstratethatSlot-MPCsuccessfullysolvescomplex,long-horizonplanningtasksthat
remainchallengingfornon-object-centricapproaches.
2

2 RelatedWork
Slot-BasedObject-CentricLearning: Slot-basedobject-centricmodelsaimtodecomposevisual
scenesintoasetofN latentembeddings,referredtoasslots,whereeachslotrepresentsadistinct
S
objectorentityinthescene[Locatelloetal.,2020,Greffetal.,2020]. Earlyapproacheslearned
suchobject-centricrepresentationsend-to-endfromsyntheticimages[Locatelloetal.,2020,Burgess
et al., 2019, Singh et al., 2022a] and videos [Kipf et al., 2022, Singh et al., 2022b, Zoran et al.,
2021], demonstrating unsupervised scene decomposition under controlled settings. More recent
methodsleveragepretrainedvisionencoders[Seitzeretal.,2023,Zadaianchuketal.,2024,Aydemir
etal.,2023]oradditionalsupervision[Elsayedetal.,2022,Baoetal.,2023]toenableobject-centric
learningoncomplexreal-worldscenes. Beyondrepresentationlearning,slot-basedrepresentations
have been shown to benefit a range of downstream task, including model-based RL [Mosbach
etal.,2025,Haramatietal.,2024],visual-question-answering[Mamaghanetal.,2025]orimitation
learning[Villar-CorralesandBehnke,2025].
Model-basedRLandMPC: Model-basedRLaimstolearnapredictivemodelofenvironment
dynamics, which can then be used to improve decision-making, either by learning policies from
imaginedexperience[Hafneretal.,2020,2025],orbyplanningactionsatinferencetime[Sutton,
1991]. Unlikemodel-freeapproaches,whichlearnreactivepoliciesdirectlyfromrewardsignals,
model-based methods explicitly reason about future state evolution, enabling improved sample
efficiencyandadaptabilitytonewsituations. However,purelypolicy-basedsolutionspresentlimited
generalizationcapabilitiesastheirlearnedpoliciesaresolelyreactive[Byravanetal.,2022]. Recent
works, such as TD-MPC2 [Hansen et al., 2024], combine policy networks with MPC for online
trajectoryoptimizationatinferencetime,demonstratingstrongperformanceforabroadvarietyof
differentenvironments. Inpractice,planningistypicallyperformedusinggradient-free,sampling-
basedoptimizationmethodssuchastheCross-EntropyMethod(CEM)[Rubinstein,1997]orModel
PredictivePathIntegralcontrol(MPPI)[Williamsetal.,2015].Recentwork[Sobaletal.,2025]shows
thepotentialofgradient-freeMPC(MPPI)withalatentdynamicsmodellearnedfromreward-free
offlinedataonasetofnavigationtaskscomparedtogoal-conditionedRL.
Object-CentricWorldModelsandRL: Object-centricworldmodelsaimtoexplicitlymodel
object dynamics and interactions in video sequences in order to forecast future object and scene
states [Villar-Corrales et al., 2023, Wu et al., 2023]. By representing scenes as compositions of
persistent entities, these approaches provide a structured abstraction that is naturally suited for
reasoninganddecision-making. Object-centricrepresentationshavebeenexploredfordownstream
control,bothinmodel-freeRL[Zadaianchuketal.,2021,Mambellietal.,2022]andmodel-based
settings [Ferraro et al., 2023, Mosbach et al., 2025, Haramati et al., 2024]. However, despite
progressinreinforcementlearning,theintegrationofslot-basedrepresentationswithdownstream
planning remains relatively underexplored. Early approaches such as O2P2 [Janner et al., 2019]
and OP3 [Veerapaneni et al., 2020] model object interactions but do not employ modern slot-
basedrepresentationsandrelyonpredefinedhigh-levelactionsratherthanlow-levelcontrolsignals.
Moreover, planning in these works is typically performed using sampling-based, gradient-free
optimizationmethods. Concurrentlytoourwork,Nametal.[2026]proposeaslot-basedworldmodel
evaluated in an MPC setting on the Push-T manipulation task. While conceptually similar, their
approachistightlycoupledtotheJEPA[LeCun,2022]architecture,doesnotmatchtheperformance
ofrecentholisticworldmodelssuchasDINO-WM[Zhouetal.,2025], andalsoreliessolelyon
sampling-basedMPC.
Gradient-basedMPC: Sampling-basedMPCmethodstypicallyevaluatehundredsorthousands
ofcandidatetrajectoriesateachdecisionstep, resultinginsubstantialcomputationalcost. When
dynamicsmodelsarerepresentedbydifferentiableneuralnetworks,itisthereforenaturaltoinstead
optimizeactionsequencesdirectlyusinggradient-basedoptimization. Theideaofgradient-based
planningdatesbacktothe1960’s[Kelley,1960],yetitssuccessfulapplicationwithlearnedworld
modelshasremainedlimited. Priorapproachesoftenrelyonlargeamountsofexpertdemonstra-
tions[Srinivasetal.,2018],areonlyevaluatedonlow-dimensionaldomains[Bharadhwajetal.,2020,
SVetal.,2023],orstruggletoscaletorealisticroboticssettings[Henaffetal.,2018]. Despitetheir
potentialbenefits,mostrecentgradient-basedMPCmethodsusinglearnedworldmodelsempirically
underperformtheirsampling-basedcounterparts,andincurhighcomputationalcostduetoalarge
numberofoptimizationiterations[Zhouetal.,2025,Parthasarathyetal.,2025,Terveretal.,2026].
3

RecentworksuchasDream-MPC[SpielerandBehnke,2026]revisitsthisdirectionbycombining
gradient-basedplanningwithlearnedrewardandvaluefunctionswithinamodel-basedreinforcement
learningframework. Incontrast,weconsideranobject-centricdynamicsmodellearnedfromoffline,
reward-freedataandformulateplanningasminimizingthedistancebetweenstructuredslotrepresen-
tations. Thisformulationremovestheneedforenvironmentinteractionduringtrainingandallows
learningtask-agnosticworldmodelsthatcanbereusedacrosstaskswithinthesameenvironment,
enablingtrajectoryoptimizationatinferencetimewithoutretraining.
3 Slot-MPC
WeproposeSlot-MPC,anovelmethodthatcombinesgradient-basedMPCwithaslot-basedobject-
centriclatentdynamicsmodel. Fig.1illustratesthemaincomponentsofourapproach,aswellasits
training(Fig.1a)andinference(Fig.1b). Slot-MPCusesaSceneParsingmodule(Section3.1)to
decomposeanimageX
t
intoobjectrepresentations,calledslotsS
t
=(s1
t
,...,sN
t
S)∈RNS×DS,where
N denotesthenumberofslotsandD theirdimensionality. Subsequently,aConditionalObject-
S S
CentricPredictor(cOCVP)autoregressivelyforecastsfutureobjectstatesoverthepredictionhorizon
H, conditionedontheinitialparsedobjectslotsandanactionsequence, whichcanberandomly
initializedorproducedbyalearnedpolicy(Section3.2). Givenagoalimage,theslotspredicted
atthefinalrolloutsteparecomparedwiththegoalslots–obtainedbyparsingthegoalimage–in
ordertooptimizetheactionsequenceviatheMPCobjectivedefinedinEq.(9)(Section3.3).
3.1 Object-CentricLatentDynamicsLearning
Slot-MPCbuildsuponSAVi[Kipfetal.,2022],arecursiveencoder–decodermodelthatparseseach
frame of a video sequence into temporally aligned object representations. At time step t, SAVi
encodestheinputframeX
t
intoN permutation-equivariantobjectembeddingsS
t
∈RNS×DS,and
uses Slot Attention [Locatello et al., 2020] to iteratively refine the previous slot representations
conditionedonimagefeaturesh
t
∈RL×Dh,whereLdenotesthenumberofspatialfeaturelocations
andD theirdimensionality. Specifically,SlotAttentionperformscross-attentionbetweenslotsand
h
imagefeatures,withattentioncoefficientsnormalizedovertheslotdimensioninordertoencourage
slotstocompeteforrepresentingfeaturelocations:
(cid:18)
q(S )·k(h
)T(cid:19)
A=softmax
NS
t−√1
D
t ∈RNS×L, (1)
S
wherekandqarelearnedlinearprojections. Theslotsarethenindependentlyupdatedviaashared
GatedRecurrentUnit(GRU)[Choetal.,2014]followedbyaresidualMulti-LayerPerceptron(MLP):
A
S =GRU(A·v(h ),S ), A = n,l , (2)
t t t−1 n,l (cid:80)L−1A
i=0 n,i
wherevisalearnedlinearprojection. ThecomputationsdescribedinEqs.(1)and(2)canberepeated
multipletimeswithsharedweightstoiterativelyrefinetheslotrepresentations,producinganaccurate
object-centricrepresentationofthescene.
To reconstruct images from slots, SAVi employs a slot decoder module. Namely, each slot is
independentlyprocessedbyaSpatialBroadcastDecoder[Wattersetal.,2019](D )toproducean
SAVi
objectimageandmask,whichcanbenormalizedandcombinedviaaweightedsumtosynthesizea
videoframe:
on,mn =D (sn), ∀sn ∈ S , (3)
t t SAVi t t t
Xˆ = (cid:88)
NS
on·m˜n with m˜n =softmax (mn). (4)
t t t t NS t
n=1
SAViistrainedself-supervisedusinganimagereconstructionloss:
T
(cid:88)
L = ||D (E (X ))−X ||2, (5)
SAVi SAVi SAVi t t 2
t=1
whereE andD correspondtothesceneparsingandobjectrenderingmodules,respectively.
SAVi SAVi
4

InspiredbyOCVP[Villar-Corralesetal.,2023],weadoptatransformer-based[Vaswanietal.,2017]
latentdynamicsmodelthatautoregressivelypredictsfutureobjectsslotsconditionedonpastobject
states. Themodelleveragesself-attentiontocaptureobjectdynamicsandagent-objectinteractions
whilepreservingthepermutationequivarianceoftheslotrepresentations.
We extend the OCVP predictor to an action-conditional setting (cOCVP), enabling the model to
explicitlyaccountforcontrolinputswhenforecastingfutureobjectdynamics. Ateachtimestep,
actionvectorsa
t
∈RNa aremappedintothepredictorembeddingspacethroughalearnablelinear
projectionf andcombinedadditivelywiththeslotrepresentationstoformthepredictorinput. Given
a
pastobjectslotsandthecorrespondingactionsequence,themodelautoregressivelyestimatesthe
nextslotstate:
Sˆ =cOCVP (cid:0) S ,f (a ),...,S ,f (a ) (cid:1) . (6)
t+1 1 a 1 t a t
Startingfromasingleseedframe,thisprocessisappliedautoregressivelybyfeedingpredictedslots
backasinputs,allowingfuturestatestobegeneratedoverapredictionhorizonH.
GiventhepretrainedSAVimodel,wetraincOCVPbyminimizingacombinedobjective:
T+1
L = (cid:88) λ ·||Xˆ −X ||2 + λ ·||Sˆ −E (X )||2, (7)
cOCVP Img t t 2 Slot t SAVi t 2
t=2 (cid:124) (cid:123)(cid:122) (cid:125) (cid:124) (cid:123)(cid:122) (cid:125)
futureframeprediction joint-embeddingalignment
whereλ andλ arescalarcoefficientsthatbalancethecontributionofeachlossterm. Thefirst
Img Slot
losspenalizesframepredictionerrors,encouragingaccuratevisualforecasting,whereasthesecond
termalignspredictedslotswiththeobject-centricrepresentationsinferredfromthecorresponding
ground-truthframes,stabilizinglatentdynamicslearningandimprovingtemporalconsistency.
3.2 PolicyLearning
InitializingMPCfromrandomlysampledactionsequencesfrequentlyresultsinsuboptimalsolutions,
sincetheoptimizerlacksameaningfulstartingpointinhigh-dimensionalactionspaces. Priorwork
hasshownthatwarm-startingMPCwithaninformedinitialactionproposalcansubstantiallyimprove
convergence and optimization stability [Parmas et al., 2018, Hansen et al., 2022]. Motivated by
this observation, we learn a policy network that provides an informed initialization for the MPC
procedure.
Tothisend,wetrainapolicymodelviabehaviorcloningfromasmallsetofexpertdemonstrations.
GivenapretrainedSAViobject-centricdecompositionmodel,thepolicynetworkπ istrainedto
θ
predicttheexpertactionsa fromthecorrespondingstructuredslot-basedlatentrepresentationsS :
t t
T
(cid:88)
L = ||π (S )−a ||2. (8)
πθ θ t t 2
t=1
Whilesuchapolicymayhavelimitedgeneralizationcapabilities,wehypothesizethatitcanprovide
astronginitialactionproposal,enablingmoreefficientMPCoptimizationandfasterconvergence.
3.3 ModelPredictiveControl
Atinferencetime,weperformmodelpredictivecontrol(MPC)inthelearnedlatentobject-centric
spaceinordertooptimizeactionssequencesforreachingagoalstate. Givenagoalimage,wefirst
encodeitusingtheobject-centricencodertoobtainasetoflatentgoalslotsS representingthe
Goal
targetobjectconfiguration. Startingfromthecurrentobservationattimestept,Slot-MPCencodes
the image into latent object slots S , which serve as the initial state for planning. The learned
t
dynamics model autoregressively forecasts future latent slot states conditioned on the initial slot
representationsandacandidateactionsequencea ,producingthepredictedslotconfiguration
t:t+H−1
Sˆ atplanninghorizonH.
t+H
MPCoptimizestheactionsequencebyminimizingthedistancebetweenthelastpredictedlatentslot
configurationSˆ andthelatentgoalslotsS . Formally,theMPCobjectiveisdefinedas
t+H Goal
J =||Sˆ −S ||2. (9)
MPC t+H Goal 2
5

Ateachoptimizationiteration,MPCevaluatescandidateactionsequencesbyrollingthemoutthrough
thelearneddynamicsmodel,updatesthembasedontheirpredictedcosts,andexecutesonlythefirst
actionoftheoptimizedsequencebeforereplanningatthenexttimestep.
Sincetheorderingofobjectslotsisnotguaranteedtobeconsistentacrosstimestepsorbetween
predictedandgoalstatesforobject-centricmodels,weapplyHungarianmatchingwhencomputing
MPCcoststoensureameaningfulcomparisonbetweenobject-centriclatentstates. Ateachtimestep,
predictedobjectslotsarealignedtogoalslotsbyminimizingthepairwiseEuclideandistanceinlatent
space. TheMPCcostsarethencomputedafteralignment,usingthematchedobjectrepresentations.
WeevaluatetwodifferentMPCvariants,namelyMPPI [Williamsetal.,2015]andgradient-based
MPC.FollowingHansenetal.[2024],weadditionallyleveragealearnedpolicynetworktowarm-
starttheoptimizationbyprovidinggoodinitialactiontrajectories. Thepolicyisrolledoutoverthe
planninghorizonusingthedynamicsmodel.
MPPI: MPPI is a sampling-based MPC method that iteratively updates parameters of a time-
dependent multivariate Gaussian with diagonal covariance using importance-weighted trajectory
costs. WefollowthevariantusedbyHansenetal.[2024]. Atoptimizationiterationj,theproposal
distributionoveractionsequencesisparameterizedby(µj,σj) ,whereµj,σj ∈Rmdenote
t:t+H−1 t t
thedistributionparametersfortimesteptintheplanninghorizonH. MPPIindependentlysamples
N trajectoriesa ∼N(µj−1,(σj−1)2I)usingrolloutsgeneratedbythelearnedmodeld ,whichare
t t t θ
thenestimatedusingtheMPCobjectivedefinedinEq.(9).
Atiterationj,MPPIselectsthetop-ktrajectorieswithlowerplanningcost,andupdatestheproposal
distributionparametersµj andσj usinganimportance-weightedempiricalestimate:
(cid:118)
µj = (cid:80)k i=1 Ω i Γ⋆ i , σj = (cid:117) (cid:117) (cid:116) (cid:80)k i=1 Ω i (Γ⋆ i −µj)2 , (10)
(cid:80)k
Ω
(cid:80)k
Ω
i=1 i i=1 i
where Ω i = e−τ(J Γ ⋆ ,i ), τ is a temperature parameter controlling the sharpness of the weighting,
andΓ⋆denotestheithtop-ktrajectorycorrespondingtocostestimateJ⋆. Afterafixednumberof
i Γ
iterationsJ,theplanningprocedureisterminatedandatrajectoryissampledfromthefinalaction
proposaldistribution. Weplanateachdecisionsteptandexecuteonlythefirstaction,i.e.,weemploy
receding-horizonMPCtoproduceafeedbackpolicy. Toreducethenumberofiterationsrequiredfor
convergence,we“warmstart”trajectoryoptimizationateachsteptbyreusingtheone-stepshifted
meanµobtainedatthepreviousstep[ArgensonandDulac-Arnold,2021],butalwaysusealarge
initialvariancetoavoidlocalminima.
Gradient-basedMPC: Adifferentiablelearneddynamicsmodelenablescomputinggradientsof
theplanningobjectivewithrespecttotheactionsequencebybackpropagatingthroughtherollout,
enabling gradient-based MPC to directly optimize a single candidate action sequence instead of
samplinghundredsoftrajectoriesasinMPPI.Specifically,theactionsareoptimizedbyminimizing
theplanningobjective(Eq.(9))viagradientdescent:
a←a−η∇J . (11)
MPC
Inspired by Spieler and Behnke [2026], we use a policy network to efficiently guide the MPC
procedure, which has shown to be beneficial for stabilizing gradient-based MPC. In contrast to
model-basedRLmethods,wedonotjointlytrainthepolicyandworldmodel,butinsteadlearneach
componentindependentlyfromofflinedatasetswithoutrewardsignals.Ratherthanrelyingonlearned
rewardorvaluefunctions,wedirectlyoptimizetheobject-wisedistancebetweenpredictedandgoal
objectslotsusinglatent-spacedistancesastheplanningobjective.Thisalternativeformulationenables
visualgoal-directedplanningwithoutrequiringrewardsduringtrainingorplanning. Additionally,
insteadofsamplingmultipletrajectoriesfromthepolicynetwork,Slot-MPCdirectlyoptimizesa
singletrajectoryinitializedbythepolicy,therebyimprovingoptimizationefficiency.
4 Experiments
Toevaluatetheeffectivenessofourobject-centricplanningframework,weinvestigatethefollowing
research questions: (i) Can an object-centric world model be learned purely from pre-collected
6

offlinetrajectoriesandsubsequentlybeusedforgoal-conditionedplanning? (ii)Doobject-centric
representationsenablemoreefficientplanningcomparedtoholisticscenerepresentations? (iii)What
components are required to efficiently solve long-horizon manipulation tasks? We evaluate our
proposedmethodonfourroboticmanipulationenvironmentsandcompareagainststate-of-the-art
worldmodelstoanswerthesequestions.
4.1 EvaluationSetup
Datasets: Weevaluateourproposedapproachonfourroboticmanipulationenvironmentsadapted
fromtwodifferentbenchmarks: ButtonPressandLeverPullfromMeta-World[Yuetal.,2020],
andStack andSquarefromrobosuite[Zhuetal.,2020]. Foreachenvironment, wegeneratetwo
offlinedatasets: (i)trajectoriescollectedbyarandomexplorationagent,and(ii)asmallsetofexpert
demonstrations solving the corresponding task. The random exploration dataset is used to train
boththeobject-centricsceneparsingmodelandthestructureddynamicsmodel,whereastheexpert
demonstrationsareusedtotrainthebehaviorcloningpolicythatwarm-startsMPC.Weuseonly
visualobservationsinallenvironmentsanddonotrelyonadditionalinputssuchasproprioceptive
states. FurtherdetailsareprovidedinSectionB.
Baselines: WecompareSlot-MPCagainstestablishedbaselinesforbothonlineandofflinerein-
forcementlearning,includinggoal-conditionedbehaviorcloning(GC-BC)[Lynchetal.,2020,Ghosh
etal.,2021],Dreamer-v3[Hafneretal.,2025],andDINO-WM[Zhouetal.,2025].DINO-WMlearns
aworldmodelthatoperatesinthelatentspaceofapretrainedDINOv2[Oquabetal.,2024]encoder
usingoffline,reward-freedata. ThelearneddynamicsmodelisthenusedforplanningviaCEM,a
gradient-free,sampling-basedMPCtechnique. Goal-conditionedbehaviorcloning(GC-BC)learnsa
goal-conditionedpolicyfromreward-free,offlinedataviabehaviorcloning. Forafaircomparison,
GC-BC, DINO-WM and Slot-MPC are trained using the same offline datasets. Dreamer-v3 is a
widely used model-based RL method that jointly learns a visual world model and a policy from
rewardsignalsobtainedthroughinteractionwiththeenvironment. Sincethismethodrequiresrewards
fortraining,wedonotusetheofflinedatasets,butinsteadfollowitsintendedtrainingprotocoland
learnthepolicythroughonlineenvironmentinteractions. FurtherdetailsareprovidedinSectionD.
Successdefinition: Forafaircomparison,allmethodsareevaluatedusingthesameprotocol. At
thebeginningofeachevaluationepisode,thesimulatorstateisinitializedusingthefirststateofan
evaluationtrajectory. DINO-WM,GC-BCandSlot-MPCadditionallyreceiveagoalimagedepicting
successfultaskcompletioninthecurrentenvironment. Ateachtime-step,allmethodsreceivethe
currentvisualobservationandselectanactionaccordingtotheirpolicy. Theactionsareexecutedin
theenvironmentandthesubsequentobservationisreturned. Thisprocedureisrepeateduntileither
thegoalissuccessfullyreachedorthemaximumepisodelengthisexceeded. Ourevaluationprotocol
differs from the procedure used by DINO-WM, which only considers short randomly sampled
sub-trajectories. Instead, we evaluate full episodes, which better reflects long-horizon planning
performanceandtaskcompletion. Wereportsuccessratesacross50evaluationepisodes. Results
obtainedusingtheoriginalDINO-WMevaluationprotocolareprovidedinSectionE.1.
4.2 ComparisontoBaselines
AquantitativecomparisonofthedifferentmethodsispresentedinTab.1. Ourproposedmethod
matchestheperformanceofDreamer-v3onMeta-World,andoutperformsallbaselinesonrobosuite.
WefurtherevaluatethequalityofthepredictedobservationsfromDINO-WMandSlot-MPCusing
PSNR,SSIM[Wangetal.,2004]andLPIPS[Zhangetal.,2018]. AlthoughDINO-WMproduces
visuallyplausiblepredictions,asindicatedinTab.2,taskcompletioncompletelyfailsfortasksthat
requireplanningoverlongerhorizons.SimilartoWangetal.[2026]wefindthattheimaginedrollouts
forlonghorizonsdonotmatchtherealdynamics. Increasingthenumberofcandidatetrajectories
andoptimizationiterationsdidnothelptoovercomethisissue,likelyduetothelargesearchspace.
WefurtherincluderesultsfortheoriginalevaluationprocedureusedforDINO-WMinTab.7,which
showthatthesuccessrateinthissettingalsosignificantlydecreaseswhenremovingproprioceptive
information and considering longer horizons. For more complex tasks such as robosuite Stack,
DINO-WMfailstoachievegoal-conditiontasksuccessevenforshortplanninghorizons.
7

Table1: TasksuccesscomparisonofSlot-MPCwithstate-of-the-artbaselines. Slot-MPCmatches
Dreamer-v3onMeta-World(ButtonPressandLeverPull)andoutperformsallbaselinesonrobosuite
(StackandSquare). Besttworesultsarehighlightedboldfaceandunderlined,respectively. Wilson
95%confidenceintervalsareshowninbrackets.
SuccessRates↑
Method ButtonPress LeverPull Stack Square
Slot-MPC 0.64[0.50,0.76] 0.52[0.39,0.65] 0.42[0.29,0.56] 0.22[0.13,0.35]
DINO-WM 0.00[0.00,0.07] 0.00[0.00,0.07] 0.00[0.00,0.07] 0.00[0.00,0.07]
Dreamer-v3 0.64[0.50,0.76] 0.56[0.42,0.69] 0.30[0.19,0.44] 0.00[0.00,0.07]
GC-BC 0.54[0.40,0.67] 0.10[0.04,0.21] 0.30[0.19,0.44] 0.00[0.00,0.07]
Table2: ComparisonofDINO-WMandSlot-MPCacrossdifferentenvironmentsonimagemetrics.
ButtonPress LeverPull Stack Square
Method PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓
DINO-WM 35.84 0.990 0.0084 25.12 0.873 0.0393 26.90 0.964 0.0313 23.38 0.938 0.0418
Slot-MPC 35.92 0.977 0.0037 23.83 0.842 0.0426 32.85 0.980 0.0146 25.51 0.919 0.0354
Wehypothesizethatthislimitationarisesbecauseholisticlatentrepresentationswouldrequiresub-
stantiallyhigherstate-actioncoveragecomparedtoobject-centricrepresentations,whichintroducea
compositionalinductivebias. Asaresult,ourmodelexhibitsstrongzero-shotgeneralizationcapa-
bilitiesandovercomeslimitationscommonlyobservedinoffline-trainedpolicies,whichgeneralize
poorlyandrelyheavilyonhigh-qualitydataandbroadstate-actioncoverage[Sobaletal.,2025].
4.3 ModelAnalysis
WeconductablationstudiestoisolatethecontributionofthedifferentdesignchoicesinSlot-MPC,
includingtheobject-centricworldmodel,thepolicyprior,andtheMPCformulation. Theresults
inTab.3ashowthattheobject-centricworldmodel,thelearnedpolicynetworkandgradient-based
MPCarethemostcriticalcomponents,asremovingthemleadstothelargestperformancedrops. In
particular,wefindthattasksrequiringlongplanninghorizons(e.g. thosefromrobosuite)strongly
benefitfromagoodpolicypriortoguidetheoptimizationprocessthroughthesearchspace.
Ourexperimentsfurthershowthatgradient-basedMPCissubstantiallymoreeffectivethansampling-
basedMPCmethodssuchasMPPIintheconsideredofflinesetting, wherestate-actioncoverage
is limited. Even when using the policy prior to guide the sampling procedure, the performance
deterioratesforMPPIsignificantly. Wehypothesizethatthisisduetothedynamicsmodelbeing
queriedoutofdistribution,leadingtosuboptimaloptimization. Inthissetting,thecharacteristicsof
gradient-basedMPC,whichstaysclosertothenominalpolicydistribution,aremorebeneficialthan
thehigherdiversityofsampling-basedMPCmethods.
ForMeta-World,Slot-MPCoftenmovesvisuallyclosetothegoal,butdoesnotalwaysapplythelast
bitofforceordisplacementneededtofullypressthebutton,ormissesorslipsoffthelever. This
isparticularlydistinctwithoutapolicyprior, andisalsothereasonforminorornoperformance
improvementofMPCcomparedtousingthepolicyonly. Addingproprioceptiveinformationorusing
subgoalscouldhelptomitigatethisandfurtherimproveperformance. Forthemorecomplextasks
fromrobosuite,gradient-basedMPCimprovesperformancenotablycomparedtothepolicyonly,
highlightingtheimportanceofplanningforlong-horizonmanipulationtasks.
WealsoperformanablationstudyoftheMPCobjectivebyusingdifferentformulationsofslot-based
costfunctions. Theresults,summarizedinTab.3b,showthatperformingMPCwithacostfunction
usingthesumofsquareddistances(SSE)orcosinesimilarityofslotsperformsbest. Usingtheslot
masksm˜nisbiasedtowardshavingaccuratemodelingofthebackgroundandlargerobjectsinstead
t
ofsmallobjects. Whilethisbiascanbemitigatedbyusingthenormalizedintersectionoverunion
(IoU)betweenslotmasks,directlyoptimizingoverslotrepresentationsispreferable,sincethesame
representationspaceisalsousedduringpredictortraining.
8

Table3:AblationstudiesSlot-MPC.(a)Impactofremovingkeycomponents,includingobject-centric
representations,MPC,andpolicyinitialization,aswellasreplacinggradient-basedMPCwithMPPI.
(b)ComparisonofdifferentMPCobjectives. Besttworesultsareboldedandunderlined,respectively.
(a)Slot-MPCcomponentablation.
SuccessRates↑
| ModelVariant                     | ButtonPress |      | LeverPull | Stack Square |
| -------------------------------- | ----------- | ---- | --------- | ------------ |
| Slot-MPC                         |             | 0.64 | 0.52      | 0.42 0.22    |
| w/oobject-centricrepresentations |             | 0.62 | 0.48      | 0.20 0.04    |
| w/oMPC                           |             | 0.64 | 0.52      | 0.36 0.18    |
| w/opolicy(zerosasinitialactions) |             | 0.32 | 0.18      | 0.00 0.00    |
| w/MPPI(nogradient-basedMPC)      |             | 0.04 | 0.04      | 0.00 0.00    |
(b)MPCobjectiveablation.
SuccessRates↑
| MPCObjective                   | ButtonPress |      | LeverPull | Stack Square |
| ------------------------------ | ----------- | ---- | --------- | ------------ |
| Cosinesimilarityofalignedslots |             | 0.64 | 0.52      | 0.44 0.22    |
| SSEofalignedslots              |             | 0.64 | 0.52      | 0.42 0.22    |
| SSEofalignedslotmasks          |             | 0.58 | 0.50      | 0.30 0.10    |
Sinceinferencetimeisacriticalfactorwhendeployingamodeltoareal-worldsystemsuchasarobot,
inTab.4wereportthetimerequiredforasingleplanningsteponasingleNVIDIARTXA6000
GPU.Object-centricslotrepresentationssubstantiallyreducethelatentfeaturespacecomparedto
patch-basedmethods,suchasDINO-WM,reducingtherepresentationsizefrom#Tokens×D to
h
N ×D . Inoursetting,weusefourslotswithadimensionalityof128,resultinginalatentspace
S S
of 4×128 compared to 196×384 for DINO-WM. This corresponds to an approximately 99%
reductioninlatentdimensionalityandleadstosignificantlyfasterplanningtimes,evenwhenusing
sampling-basedMPPI.Additionally,warm-startingMPCbyusingapolicypriorfurtherimproves
computationalefficiencyofSlot-MPC.
Table4: ComparisonofplanningexecutiontimesfordifferentmethodsonasingleNVIDIARTX
A6000GPU.Slot-MPCissubstantiallymorecomputationallyefficientthanthepatch-basedbaseline.
Planningtime(s)↓
| Method                      |     | Meta-World  |     | robosuite   |
| --------------------------- | --- | ----------- | --- | ----------- |
| Slot-MPC                    |     | 0.42±0.01   |     | 0.48±0.02   |
| w/MPPI(nogradient-basedMPC) |     | 4.22±0.06   |     | 5.19±0.03   |
| DINO-WM                     |     | 144.37±0.83 |     | 145.30±0.72 |
5 Conclusion
We study the importance of object-centric representations for efficient planning with MPC and
introduceSlot-MPC,anovelmethodforperforminggradient-basedMPCwithobject-centriclatent
dynamics. Slot-MPClearnsobject-centricrepresentationsfromreward-freeofflinedataandcombines
aslot-basedworldmodelwithgradient-basedtrajectoryoptimizationforgoal-conditionedplanning
directly from visual observations. By optimizing action trajectories in a compact object-centric
latentspace,Slot-MPCsignificantlyimprovesbothtaskperformanceandcomputationalefficiency
comparedtostate-of-the-artbaselines. Ourexperimentsonsimulatedroboticmanipulationtasks
demonstratethatobject-centricrepresentationsareapowerfulinductivebiasforcontrol,particularly
forlong-horizonplanningproblems. Overall,ourresultssuggestthatstructuredobject-centricworld
models are a promising direction for scalable and efficient model-based control from raw visual
observations.
9

Acknowledgements
This work was funded by the Federal Ministry of Research, Technology and Space of Germany
(BMFTR)withintheWestAI-AIServiceCenterWest,grantno.16IS22094AandwithintheRobotics
InstituteGermany,grantno. 16ME0999. ComputationalresourceswereprovidedbytheGermanAI
ServiceCenterWestAI.
References
ArthurArgensonandGabrielDulac-Arnold. Model-basedofflineplanning. InInternationalConfer-
enceonLearningRepresentations(ICLR),2021.
GörkayAydemir,WeidiXie,andFatmaGuney. Self-supervisedobject-centriclearningforvideos. In
AdvancesinNeuralInformationProcessingSystems(NeurIPS),2023.
Zhipeng Bao, Pavel Tokmakov, Yu-Xiong Wang, Adrien Gaidon, and Martial Hebert. Object
discoveryfrommotion-guidedtokens. InIEEE/CVFConferenceonComputerVisionandPattern
Recognition(CVPR),2023.
HomangaBharadhwaj,KevinXie,andFlorianShkurti. Model-predictivecontrolviacross-entropy
andgradient-basedoptimization. InConferenceonLearningforDynamicsandControl(L4DC),
2020.
Christopher P Burgess, Loic Matthey, Nicholas Watters, Rishabh Kabra, Irina Higgins, Matt
Botvinick,andAlexanderLerchner. Monet:Unsupervisedscenedecompositionandrepresentation.
arXiv:1901.11390,2019.
ArunkumarByravan,LeonardHasenclever,PiotrTrochim,MehdiMirza,AlessandroDavideIalongo,
Yuval Tassa, Jost Tobias Springenberg, Abbas Abdolmaleki, Nicolas Heess, Josh Merel, and
MartinA.Riedmiller. Evaluatingmodel-basedplanningandplanneramortizationforcontinuous
control. InInternationalConferenceonLearningRepresentations(ICLR),2022.
KyunghyunCho,BartvanMerriënboer,CaglarGulcehre,DzmitryBahdanau,FethiBougares,Holger
Schwenk, and Yoshua Bengio. Learning phrase representations using RNN encoder–decoder
for statistical machine translation. In Conference on Empirical Methods in Natural Language
Processing(EMNLP),2014.
C.R.CutlerandB.L.Ramaker. Dynamicmatrixcontrol-Acomputercontrolalgorithm. IEEE
TransactionsonAutomaticControl(TAC),17:72,1979.
GamaleldinElsayed,AravindhMahendran,SjoerdVanSteenkiste,KlausGreff,MichaelCMozer,
andThomasKipf. SAVi++: Towardsend-to-endobject-centriclearningfromreal-worldvideos. In
AdvancesinNeuralInformationProcessingSystems(NeurIPS),2022.
StefanoFerraro,PietroMazzaglia,TimVerbelen,andBartDhoedt. Focus: Object-centricworldmod-
elsforroboticsmanipulation. InAdvancesinNeuralInformationProcessingSystemsWorkshops
(NeurIPSw),2023.
DibyaGhosh,AbhishekGupta,AshwinReddy,JustinFu,ColineManonDevin,BenjaminEysenbach,
and Sergey Levine. Learning to reach goals via iterated supervised learning. In International
ConferenceonLearningRepresentations(ICLR),2021.
KlausGreff,SjoerdVanSteenkiste,andJürgenSchmidhuber. Onthebindingprobleminartificial
neuralnetworks. arXiv:2012.05208,2020.
DavidHaandJürgenSchmidhuber. Worldmodels. arXiv:1803.10122,2018.
Danijar Hafner, Timothy P. Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to control:
Learningbehaviorsbylatentimagination.InInternationalConferenceonLearningRepresentations
(ICLR),2020.
DanijarHafner,JurgisPasukonis,JimmyBa,andTimothyLillicrap. Masteringdiversecontroltasks
throughworldmodels. Nature,640:647–653,2025.
10

Nicklas Hansen, Hao Su, and Xiaolong Wang. TD-MPC2: Scalable, robust world models for
continuouscontrol. InInternationalConferenceonLearningRepresentations(ICLR),2024.
NicklasAHansen,HaoSu,andXiaolongWang. Temporaldifferencelearningformodelpredictive
control. InInternationalConferenceonMachineLearning(ICML),2022.
DanHaramati,TalDaniel,andAvivTamar. Entity-centricreinforcementlearningforobjectmanipu-
lationfrompixels. InInternationalConferenceonLearningRepresentations(ICLR),2024.
Mikael Henaff, William F. Whitney, and Yann LeCun. Model-based planning with discrete and
continuousactions. arXiv:1705.07177,2018.
Michael Janner, Sergey Levine, William T. Freeman, Joshua B. Tenenbaum, Chelsea Finn, and
JiajunWu. Reasoningaboutphysicalinteractionswithobject-orientedpredictionandplanning. In
InternationalConferenceonLearningRepresentations(ICLR),2019.
DanielKahneman,AnneTreisman,andBrianJGibbs. Thereviewingofobjectfiles: Object-specific
integrationofinformation. Cognitivepsychology,24(2):175–219,1992.
HenryJ.Kelley. Gradienttheoryofoptimalflightpaths. ARSJournal,30(10):947–954,1960.
DiederikP.KingmaandJimmyBa. Adam: Amethodforstochasticoptimization. InInternational
ConferenceonLearningRepresentations(ICLR),2015.
Thomas Kipf, Gamaleldin F. Elsayed, Aravindh Mahendran, Austin Stone, Sara Sabour, Georg
Heigold,RicoJonschkowski,AlexeyDosovitskiy,andKlausGreff. Conditionalobject-centric
learningfromvideo. InInternationalConferenceonLearningRepresentations(ICLR),2022.
YannLeCun. Apathtowardsautonomousmachineintelligenceversion0.9.2,2022-06-27,2022.
URLhttps://openreview.net/pdf?id=BZ5a1r-kVsf.
FrancescoLocatello,DirkWeissenborn,ThomasUnterthiner,AravindhMahendran,GeorgHeigold,
JakobUszkoreit,AlexeyDosovitskiy,andThomasKipf. Object-centriclearningwithslotattention.
InAdvancesinNeuralInformationProcessingSystems(NeurIPS),2020.
Corey Lynch, Mohi Khansari, Ted Xiao, Vikash Kumar, Jonathan Tompson, Sergey Levine, and
PierreSermanet. Learninglatentplansfromplay. InConferenceonRobotLearning(CoRL),2020.
Amir Mohammad Karimi Mamaghan, Samuele Papa, Karl Henrik Johansson, Stefan Bauer, and
AndreaDittadi. Exploringtheeffectivenessofobject-centricrepresentationsinvisualquestion
answering:Comparativeinsightswithfoundationmodels.InInternationalConferenceonLearning
Representations(ICLR),2025.
Davide Mambelli, Frederik Träuble, Stefan Bauer, Bernhard Schölkopf, and Francesco Lo-
catello. Compositional multi-object reinforcement learning with linear relation networks.
arXiv:2201.13388,2022.
AjayMandlekar,SoroushNasiriany,BowenWen,IretiayoAkinola,YashrajNarang,LinxiFan,Yuke
Zhu,andDieterFox. Mimicgen: Adatagenerationsystemforscalablerobotlearningusinghuman
demonstrations. InConferenceonRobotLearning(CoRL),2023.
Malte Mosbach, Jan Niklas Ewertz, Angel Villar-Corrales, and Sven Behnke. Sold: Slot object-
centriclatentdynamicsmodelsforrelationalmanipulationlearningfrompixels. InInternational
ConferenceonMachineLearning(ICML),2025.
HeejeongNam,QuentinLeLidec,LucasMaes,YannLeCun,andRandallBalestriero. Causal-JEPA:
Learningworldmodelsthroughobject-levellatentinterventions. arXiv:2602.11389,2026.
MaximeOquab,TimothéeDarcet,ThéoMoutakanni,HuyV.Vo,MarcSzafraniec,VasilKhalidov,
PierreFernandez,DanielHAZIZA,FranciscoMassa,AlaaeldinEl-Nouby,MidoAssran,Nicolas
Ballas,WojciechGaluba,RussellHowes,Po-YaoHuang,Shang-WenLi,IshanMisra,Michael
Rabbat,VasuSharma,GabrielSynnaeve,HuXu,HerveJegou,JulienMairal,PatrickLabatut,Ar-
mandJoulin,andPiotrBojanowski. DINOv2: Learningrobustvisualfeatureswithoutsupervision.
TransactionsonMachineLearningResearch(TMLR),2024.
11

PaavoParmas,CarlEdwardRasmussen,JanPeters,andKenjiDoya. PIPPS:Flexiblemodel-based
policy search robust to the curse of chaos. In International Conference on Machine Learning
(ICML),2018.
ArjunParthasarathy,NimitKalra,RohunAgrawal,YannLeCun,OumaymaBounou,PavelIzmailov,
andMicahGoldblum. Closingthetrain-testgapinworldmodelsforgradient-basedplanning.
arXiv:2512.09929,2025.
Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary DeVito,
Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. Automatic differentiation in
PyTorch. In International Conference on Neural Information Processing Systems Workshops
(NeurIPSw),2017.
MichaelPsenka,MichaelRabbat,AditiKrishnapriyan,YannLeCun,andAmirBar.Parallelstochastic
gradient-basedplanningforworldmodels. arXiv:2602.00475,2026.
JacquesRichalet,AndréRault,Jean-LouisTestud,andJeanPapon.Modelpredictiveheuristiccontrol.
Automatica,14(5):413–428,1978.
ReuvenY.Rubinstein. Optimizationofcomputersimulationmodelswithrareevents. European
JournalofOperationalResearch(EJOR),1997.
JyothirSV,SiddharthaJalagam,YannLeCun,andVladSobal. Gradient-basedplanningwithworld
models. arXiv:2312.17227,2023.
MaximilianSeitzer,MaxHorn,AndriiZadaianchuk,DominikZietlow,TianjunXiao,Carl-Johann
Simon-Gabriel,TongHe,ZhengZhang,BernhardSchölkopf,ThomasBrox,etal. Bridgingthe
gaptoreal-worldobject-centriclearning. InInternationalConferenceonLearningRepresentations
(ICLR),2023.
GautamSingh,FeiDeng,andSungjinAhn. IlliterateDALL-Elearnstocompose. InInternational
ConferenceonLearningRepresentations(ICLR),2022a.
GautamSingh,Yi-FuWu,andSungjinAhn. Simpleunsupervisedobject-centriclearningforcomplex
andnaturalisticvideos. InAdvancesinNeuralInformationProcessingSystems(NeurIPS),2022b.
Vlad Sobal, Wancong Zhang, Kynghyun Cho, Randall Balestriero, Tim G. J. Rudner, and Yann
LeCun. Learningfromreward-freeofflinedata: Acaseforplanningwithlatentdynamicsmodels.
InAdvancesinNeuralInformationProcessingSystems(NeurIPS),2025.
Jonathan Spieler and Sven Behnke. Dream-MPC: Gradient-based model predictive control with
latentimagination. InInternationalConferenceonMachineLearning(ICML),2026.
Aravind Srinivas, Allan Jabri, Pieter Abbeel, Sergey Levine, and Chelsea Finn. Universal plan-
ningnetworks: Learninggeneralizablerepresentationsforvisuomotorcontrol. InInternational
ConferenceonMachineLearning(ICML),2018.
Richard S. Sutton. Dyna, an integrated architecture for learning, planning, and reacting. ACM
SIGARTBulletin,2(4):160–163,1991.
BasileTerver,Tsung-YenYang,JeanPonce,AdrienBardes,andYannLeCun. Whatdrivessuccess
inphysicalplanningwithjoint-embeddingpredictiveworldmodels? arXiv:2512.24497,2026.
AshishVaswani,NoamShazeer,NikiParmar,JakobUszkoreit,LlionJones,AidanNGomez,Łukasz
Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information
ProcessingSystems(NeurIPS),2017.
RishiVeerapaneni,JohnD.Co-Reyes,MichaelChang,MichaelJanner,ChelseaFinn,JiajunWu,
JoshuaTenenbaum,andSergeyLevine. Entityabstractioninvisualmodel-basedreinforcement
learning. InConferenceonRobotLearning(CoRL),2020.
AngelVillar-CorralesandSvenBehnke. PlaySlot: Learninginverselatentdynamicsforcontrollable
object-centricvideopredictionandplanning. InInternationalConferenceonMachineLearning
(ICML),2025.
12

Angel Villar-Corrales, Ismail Wahdan, and Sven Behnke. Object-centric video prediction via
decoupling of object dynamics and interactions. In IEEE International Conference on Image
Processing(ICIP),2023.
AngelVillar-Corrales,GjergjPlepi,andSvenBehnke. TextOCVP:Object-centricvideoprediction
withlanguageguidance. TransactionsonMachineLearningResearch(TMLR),2026.
YingWang,OumaymaBounou,GaoyueZhou,RandallBalestriero,TimG.J.Rudner,YannLeCun,
andMengyeRen. Temporalstraighteningforlatentplanning. arXiv:2603.12231,2026.
ZhouWang, AlanCBovik, HamidRSheikh, andEeroPSimoncelli. Imagequalityassessment:
Fromerrorvisibilitytostructuralsimilarity. IEEETransactionsonImageProcessing(TIP),13(4):
600–612,2004.
NicholasWatters,LoicMatthey,ChristopherPBurgess,andAlexanderLerchner.Spatialbroadcastde-
coder: AsimplearchitectureforlearningdisentangledrepresentationsinVAEs. arXiv:1901.07017,
2019.
Grady Williams, Andrew Aldrich, and Evangelos A. Theodorou. Model predictive path integral
controlusingcovariancevariableimportancesampling. arXiv:1509.01149,2015.
ZiyiWu,NikitaDvornik,KlausGreff,ThomasKipf,andAnimeshGarg. SlotFormer: Unsupervised
visualdynamicssimulationwithobject-centricmodels. InInternationalConferenceonLearning
Representations(ICLR),2023.
TianheYu,DeirdreQuillen,ZhanpengHe,RyanJulian,KarolHausman,ChelseaFinn,andSergey
Levine. Meta-World: Abenchmarkandevaluationformulti-taskandmetareinforcementlearning.
InConferenceonRobotLearning(CoRL),2020.
AndriiZadaianchuk,MaximilianSeitzer,andGeorgMartius. Self-supervisedvisualreinforcement
learningwithobject-centricrepresentations. InInternationalConferenceonLearningRepresenta-
tions(ICLR),2021.
AndriiZadaianchuk,MaximilianSeitzer,andGeorgMartius. Object-centriclearningforreal-world
videosbypredictingtemporalfeaturesimilarities. InAdvancesinNeuralInformationProcessing
Systems(NeurIPS),2024.
ChuhanZhang,AnkushGupta,andAndrewZisserman. Isanobject-centricvideorepresentation
beneficialfortransfer? InAsianConferenceonComputerVision(ACCV),2022.
RichardZhang,PhillipIsola,AlexeiAEfros,EliShechtman,andOliverWang. Theunreasonable
effectivenessofdeepfeaturesasaperceptualmetric. InIEEEConferenceonComputerVisionand
PatternRecognition(CVPR),2018.
Gaoyue Zhou, Hengkai Pan, Yann LeCun, and Lerrel Pinto. DINO-WM: World models on pre-
trainedvisualfeaturesenablezero-shotplanning.InInternationalConferenceonMachineLearning
(ICML),2025.
Yuke Zhu, Josiah Wong, Ajay Mandlekar, Roberto Martín-Martín, Abhishek Joshi, Kevin Lin,
SoroushNasiriany,andYifengZhu. robosuite: Amodularsimulationframeworkandbenchmark
forrobotlearning. arXiv:2009.12293,2020.
DanielZoran,RishabhKabra,AlexanderLerchner,andDaniloJRezende. PARTS:Unsupervised
segmentationwithslots,attentionandindependencemaximization. InIEEE/CVFInternational
ConferenceonComputerVision(ICCV),2021.
13

Appendix
A LimitationsandFutureWork 14
B DatasetsandSimulationEnvironments 14
C ImplementationDetails 15
C.1 Object-CentricLearningandWorldModeling . . . . . . . . . . . . . . . . . . . . . . 15
C.2 PolicyModel . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
C.3 TrainingDetails . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
C.4 ModelPredictiveControl . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
D Baselines 17
E AdditionalResults 17
E.1 DINO-WMEvaluationProcedure . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
E.2 QualitativeResults . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
A LimitationsandFutureWork
Slot-MPCrelieslikeDINO-WMongoalimagesforplanningandongroundtruthactionsfortraining
thedynamicsmodelandpolicy,whichbothmaynotalwaysbeavailable. Anextensionofthiswork
couldinvolvetextconditioning[Villar-Corralesetal.,2026],whichallowsforspecifyingthegoal
usingnaturallanguage. Performancedependsonthequalityoftheobject-centricdecomposition
modelandthepolicyprior. Furtherimprovingperformancebyusingsubgoalsandextendingthe
frameworktoreal-worldroboticsettingsispartoffuturework,potentiallyleveragingmorecapable
decompositionmodelssuchasDINOSAUR[Seitzeretal.,2023].
B DatasetsandSimulationEnvironments
WeevaluateSlot-MPCandbaselinemodelsonfourtasksfromdifferentmanipulationbenchmarks.
Imageinputsareresizedtoaresolutionof64×64forSlot-MPCandDreamer-v3,and224×224for
DINO-WM.TheenvironmentsareillustratedinFig.2.
ButtonPress LeverPull Stack Square
Figure2: Environments. WeevaluateSlot-MPConfourdifferentenvironments.
14

Meta-World[Yuetal.,2020]: isanopensourcebenchmark(MITlicense)containingcontinuous
controlroboticmanipulationenvironments. WeconsidertheButtonPresstask,whichrequiresthe
robot to press a button that is randomly positioned in the scene, and the Lever Pull task, where
therobotneedstopullaleverwhosepositionisrandomlyinitialized. AlltasksfromMeta-World
sharethesameembodiment,observationspace(dim(S) = 39)andactionspace(dim(A) = 4). We
refertoYuetal.[2020]forthedefinitionsoftherewardfunctionsandsuccessmetricsusedinthe
Meta-Worldtasks. Wegenerateatrainingdatasetconsistingof9,000trainingsequencesand1,000
validationtrajectoriesusingarandomexplorationpolicy. Weusetheprovidedexpertpoliciesfrom
Meta-Worldtogenerateasmalltrainingsetof200expertdemonstrations,aswellasanevaluation
datasetcontaining50successfulsequences.
robosuite[Zhuetal.,2020]: WefurtherconsidertheStacktaskfromrobosuite,wherethegoalisto
stacktwoblocksontopofeachother. Theblocksarerandomlypositionedonthetable. Additionally,
weconsidertheSquare_D1taskfromMimicGen,wheretherobotneedstopickasquarenutandplace
itonarod.Thepositionsofthenutandrodarerandomlyinitialized.Theactiondimensionoftheused
Pandarobotisseven. Wegenerateadatasetcontaining10,000sequences,splitinto9,000training
and1,000validationsequencesusingarandomexplorationpolicy. WeuseMimicGen[Mandlekar
etal.,2023]togenerate2,000expertdemonstrationsforbehaviorcloningandevaluation, which
aresplitinto1,800sequencesfortrainingthepolicyand200sequencesforevaluation. MimicGen
requiresusinganOSC_POSEcontroller,whichcontrolstheend-effectorpose. robosuiteislicensed
underanMITlicenseandMimicGenunderanNVIDIAlicense.
C ImplementationDetails
Inthissection,wedescribethenetworkarchitectureandtrainingdetailsforeachofthecomponents
ofSlot-MPC.OurmodelsareimplementedinPyTorch[Paszkeetal.,2017]andaretrainedona
singleNVIDIARTXA6000GPU.
C.1 Object-CentricLearningandWorldModeling
WecloselyfollowVillar-CorralesandBehnke[2025]fortheimplementationofboththeobject-centric
decompositionmodelandthestructureddynamicsmodel.
Object-CentricDecomposition: Theobject-centricdecompositionisbasedonSAVi[Kipfetal.,
2022],arecursiveslot-basedmodelthatservesasoursceneparsingandobjectrenderingmodules.
Specifically, we adopt their proposed CNN-based image encoder E and slot decoder D ,
SAVi SAVi
as well as their transformer-based transition module, and Slot Attention corrector. We use four
128-dimensionalobjectslotsforalltheevaluateddataset,whichsufficetoseparatetheagent,the
differentobjects,andthebackground. Foralldatasets,theinitialslotrepresentationsS arerandomly
0
initializedandoptimizedaslearnableparametersviabackpropagation.Furthermore,weusethreeSlot
Attentioniterationsonthefirstobservationtoensureastableinitialdecomposition. Forsubsequent
frames,asingleiterationsufficestorecursivelyrefinetheslotrepresentationsconditionedonthe
newlyobservedimagefeatures.
Object-CentricWorldModeling: OurcOCVPstructuredworldmodelisanobject-centrictrans-
formerpredictorinspiredbyVillar-Corralesetal.[2023],Wuetal.[2023]. ThecOCVPmodule
consistsoffourtransformerlayerswith256-dimensionaltokens,eightattentionheadsofdimension
64,andafeed-forwardhiddendimensionof1024.
Toenableaction-conditionedprediction, cOCVPmapsboththeactionsa andobjectslotsS
1:t 1:t
intoasharedtokenembeddingspaceusinglearnableprojectionlayers. Theprojectedobjectslots
arethenconditionedbyaddingthecorrespondingprojectedactionateachtime-step. Furthermore,
followingWuetal.[2023],weaugmentthetokenswithatemporalsinusoidalpositionalembedding,
whichassignsthesameencodingtoalltokensfromthesametime-step,thuspreservingtheinherent
permutationequivarianceoftheobjects.
15

C.2 PolicyModel
Thepolicymodelπ isatransformerthatjointlyprocessestheobjectsslotsfromasingletimestep
θ
S togetherwithanadditionallearnableactionembedding[ACT]inordertoregressanexpertaction.
t
Themodelconsistsoffourtransformerlayerswith256-dimensionaltokens,four64-dimensional
headsandafeed-forwardhiddendimensionof1024. Throughtheattentionmechanism,information
fromtheobjectslotsisaggregatedintothe[ACT]token,whichissubsequentlymappedtoproducea
singleactionˆausingalearnablelinearprojectionhead. Thepolicyisrolledoutautoregressivelyover
thehorizonH usingthelearneddynamicsmodeltogenerateaninitialactionsequenceforMPC.
C.3 TrainingDetails
SAVi Training: SAVi is trained for object-centric decomposition using the Adam opti-
mizer [Kingma and Ba, 2015], a batch size of 64, sequences of length eight frames, and a base
learning rate of 10−4, which is linearly warmed-up for the first 4000 steps, followed by cosine
annealingfortheremainingofthetrainingprocess. Moreover,weclipthegradientstoamaximum
normof0.05.
cOCVPTraining: WetrainourcOCVPmodulegivenapretrainedSAVidecompositionmodel.
ThismoduleistrainedwiththeAdamoptimizer[KingmaandBa,2015],batchsizeof64,andabase
learningrateof2×10−4, whichdecreasesduringtrainingwithacosineannealingschedule. To
stabilizethetraining,weclipthegradientstoamaximumnormof0.05. Wesetthelossweightsto
λ =1,andλ =1.
Img Slot
π Training: Wetraintheπ modulegivenpretrainedandfrozenSAVi,andcOCVPmodules. This
θ θ
moduleistrainedwiththeAdamoptimizer[KingmaandBa,2015],batchsizeof64,andalearning
rateof3×10−4.
C.4 ModelPredictiveControl
WecomparetwodifferentMPCmethods: gradient-basedMPCandMPPI.Forboth,weuseapolicy
networktowarm-starttheoptimizationandclipactionstothevalidactionsbounds.
Table5: Gradient-basedMPCparameters.
Hyperparameter Value
HorizonH 8(Square,Meta-World)
15(Stack)
Iterations 3
Numberofsamples 1
Policypriorsamples 1
Stepsizeη 0.001
Table6: MPPIparameters.
Hyperparameter Value
HorizonH 15
Iterations 5
Numberofsamples 64
Numberofelites 16
Policypriorsamples 16
Minimumstd. 0.05
Maximumstd. 2.0
Temperature 1.0
16

D Baselines
Dreamer-v3: ForDreamer-v3,weusetheofficialreimplementationfromhttps://github.com/
danijar/dreamerv3,whichislicensedunderanMITlicense. Forafaircomparison,wedonot
usetheofflinedataset, buttraintheagentthroughinteractionwiththeenvironmentfor1Msteps
forMeta-World,and5Mstepsforrobosuite. Wefollowtheauthors’suggestedhyperparametersfor
visualobservations(DMControl)anduseanupdate-to-data(UTD)ratioof256forallenvironments.
Weuseamodelsizeof12MparametersforMeta-Worldand25Mparametersforrobosuite. Please
refertoHafneretal.[2025]foracompletelistofhyperparameters.
DINO-WM: For DINO-WM, we use the official implementation provided by the authors and
licensedunderanMITlicense: https://github.com/gaoyuezhou/dino_wm. Weusethedefault
hyperparameterssuggestedbytheauthors.
OCVP: For our experiments, we use OCVP as the object-centric world model and base our
implementation on the official implementation of PlaySlot [Villar-Corrales and Behnke, 2025]:
https://github.com/angelvillar96/PlaySlot.
Non-Object-CentricBaseline: Thisbaselinemodelfollowsthesamegeneralframeworkasour
proposedmodel,butreplacestheobject-centricSAViencoderanddecoderwithasimpleconvolutional
auto-encoderwhilekeepingtheothermodulesunchanged;thusallowingustoablatetheeffectof
object-centricrepresentationsforMPCcomparedtoasingleholisticlatent.
E AdditionalResults
E.1 DINO-WMEvaluationProcedure
WefurtherprovideresultswhenevaluatingDINO-WMusingtheprocedureusedbytheauthors,i.e.,
samplingrandomsubtrajectoriesoflengthH = 25fromtheevaluationdatasetandusingthelast
observationforplanning. Successisdefinedasreachingastatethatmatchesthegoalstateupto
somethreshold. Wefindthatremovingproprioceptiveinformationleadstoasignificantdecreaseof
thesuccessrate. Ourexperimentsshowthatwithincreasingsubtrajectorylength,thesuccessrate
significantlydropsevenwhensignificantlyincreasingthenumberofcandidatetrajectories,aligning
withasuccessrateofzeroobtainedwhenevaluatingthecompletetrajectoryfollowingourevaluation
procedure. Recentworksalsofindthatremovingproprioceptiveinformation[Terveretal.,2026]and
usingalongergoalhorizon[Parthasarathyetal.,2025,Psenkaetal.,2026,Wangetal.,2026]both
leadtoasignificantdropofthesuccessrateforDINO-WM.
ForMeta-Worldandrobosuite,weconsiderataskassuccessfulifthedistancebetweentheactual
systemstateandthedesiredgoalstate(intheoriginalstatespace)islessthanapredefinedthreshold.
Thisthresholdissetto0.3forMeta-Worldand1.0forrobosuite.
Table7: DINO-WMevaluationresultsusingtheprocedureproposedbytheauthors.
(Sub-)GoalReachingRate↑
Model ButtonPress LeverPull Stack Square
Slot-MPC 0.80 0.86 0.38 0.18
DINO-WM 0.56 0.10 0.00 0.04
w/goalhorizonH =50 0.28 0.00 0.00 0.00
E.2 QualitativeResults
Fig. 3 shows that while the predicted rollouts of DINO-WM are quite accurate for smaller goal
horizons,thepredictionsdeviatefromtheactualenvironmentrolloutforlongergoalhorizons. As
showninFig.3b,DINO-WMpredictsthatthebuttonispressed(bottomrow),whiletherobotactually
missesthebuttonbybeingtoofarrightofit(toprow).
17

(a)GoalhorizonH=25
(b)GoalhorizonH=50
Figure3: DINO-WMevaluationonsubtrajectories. (a)WithagoalhorizonofH=25. (b)Witha
goalhorizonofH=50. Thebottomrowcorrespondstothepredictedframesandthetoprow(shaded
forvisualdistinction)aretheactualobservationsfromthesimulator. Thelastimageisthegoalimage.
Wevisualizethepredictionsanddecompositionresultsoftheobject-centricmodelsofSlot-MPC
fortheconsideredenvironments. Figs.4to7showthatscenesareparsedintomeaningfulobject
representations.
Figure4: PredictionsanddecompositionresultsforButtonPress. Slot-MPCassignsaslotforthe
background,aslotfortherobotarm,andaslotforthebutton.
—Appendicescontinueonnextpage—
18

Figure 5: Predictions and decomposition results for Lever Pull. Slot-MPC assigns a slot for the
background,aslotfortherobotarm,andaslotforthelever.
Figure6:PredictionsanddecompositionresultsforStack.Slot-MPCassignsaslotforthebackground,
aslotfortherobot’sgripper,andaslotforeachcube.
Figure7: PredictionsanddecompositionresultsforSquare. Slot-MPCassignsaslotfortheback-
ground,aslotfortherobot’sgripper,aslotforthepeg,andaslotforthenut.
19