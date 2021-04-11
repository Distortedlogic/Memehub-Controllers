from pathlib import Path
from typing import Dict, List, cast

import arrow
from arrow.arrow import Arrow
from decouple import config

SECRET_KEY = cast(str, config("SECRET_KEY"))
FLASK_ENV = cast(str, config("FLASK_ENV"))
PROD = FLASK_ENV == "production"

LOAD_MEME_CLF_VERSION = "0.4.4"
LOAD_STONK_VERSION = "0.0"
MEME_CLF_VERSION = "0.4.5"
STONK_VERSION = "0.0"
NOT_A_MEME_VERSION = "0.0.1"

MODELS_REPO = "src/models/"
NOT_MEME_REPO = "src/data/not_a_meme/"
NOT_TEMPLATE_REPO = "src/data/not_a_template/"
MEMES_REPO = "src/data/memes/"
INCORRECT_REPO = "src/data/incorrect/"
BLANKS_REPO = "src/data/blanks/"
MEME_CLF_REPO = MODELS_REPO + "market/" + MEME_CLF_VERSION + "/meme_clf/{}/"
STONK_REPO = (
    MODELS_REPO + "market/" + MEME_CLF_VERSION + "/stonks/" + STONK_VERSION + "/{}/"
)
LOAD_MEME_CLF_REPO = MODELS_REPO + "market/" + LOAD_MEME_CLF_VERSION + "/meme_clf/{}/"
LOAD_STONK_REPO = (
    MODELS_REPO
    + "market/"
    + LOAD_MEME_CLF_VERSION
    + "/stonks/"
    + LOAD_STONK_VERSION
    + "/{}/"
)
NOT_A_MEME_MODEL_REPO = MODELS_REPO + "not_a_meme/" + NOT_A_MEME_VERSION + "/{}/"
LOAD_STATIC_PATH = MODELS_REPO + "market/{}/"


def backup(path: str):
    return path.replace("src/models/", "src/backup/")


Path(NOT_MEME_REPO).mkdir(parents=True, exist_ok=True)  # type: ignore
Path(NOT_TEMPLATE_REPO).mkdir(parents=True, exist_ok=True)  # type: ignore
Path(MEMES_REPO).mkdir(parents=True, exist_ok=True)  # type: ignore
Path(INCORRECT_REPO).mkdir(parents=True, exist_ok=True)  # type: ignore
Path(BLANKS_REPO).mkdir(parents=True, exist_ok=True)  # type: ignore

for folder in ["reg", "jit", "cp"]:
    Path(MEME_CLF_REPO.format(folder)).mkdir(parents=True, exist_ok=True)
    Path(STONK_REPO.format(folder)).mkdir(parents=True, exist_ok=True)
    Path(NOT_A_MEME_MODEL_REPO.format(folder)).mkdir(parents=True, exist_ok=True)
    Path(backup(MEME_CLF_REPO).format(folder)).mkdir(parents=True, exist_ok=True)
    Path(backup(STONK_REPO).format(folder)).mkdir(parents=True, exist_ok=True)
    Path(backup(NOT_A_MEME_MODEL_REPO).format(folder)).mkdir(
        parents=True, exist_ok=True
    )


LOGS_PATH = "src/logs/"
Path(LOGS_PATH).mkdir(parents=True, exist_ok=True)  # type: ignore
LOG_FILES = ["market.log"]
for file in LOG_FILES:
    Path(LOGS_PATH + file).touch(exist_ok=True)

MONTH_TD = 60 * 60 * 24 * 30
WEEK_TD = 60 * 60 * 24 * 7
DAY_TD = 60 * 60 * 24
HOUR_TD = 60 * 60

IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNEL = 3
BATCH_SIZE = 32


def get_beginning() -> int:
    return (
        cast(Arrow, arrow.utcnow())
        .shift(days=-31)
        .replace(hour=0, minute=0, second=0)
        .timestamp
    )


FULL_SUB_LIST = ["dankmemes", "memes"]

PUSHSHIFT_URI = r"https://api.pushshift.io/reddit/search/submission?subreddit={}&after={}&before={}&size={}"


IMGFLIP_TEMPALTE_URI = "https://imgflip.com/memetemplates?page={}"

USER_TEMPLATES: List[Dict[str, str]] = [
    dict(
        name="sad linus",
        blank_url="https://i.imgflip.com/3n7e0h.png",
        page="https://imgflip.com/meme/220374449/Sad-Linus?page={}",
    ),
    dict(
        name="confused unga bunga",
        blank_url="https://i.imgflip.com/38z8jt.jpg",
        page="https://imgflip.com/meme/196479497/Confused-Unga-Bunga?page={}",
    ),
]

DONT_USE_TEMPLATES = set(
    [
        "gasp rage face",
        "baby crying",
        "if you know what i mean bean",
        "slenderman",
        "brian williams was there 3",
        "permission bane",
        "tui",
        "dancing trollmom",
        "what my friends think i do",
        "no soup for you",
        "speechless colbert face",
        "angry baby",
        "valentines day card meme",
        "blank colored background",
        "blank comic panel 2x2",
        "reverse kalm panik",
        "does your dog bite",
        "dumpster fire",
        "wants to know your location",
        "gangnam style",
        "mom can we have",
        "mr bean",
        "smiling cat",
        "noooo you cant just",
        "be like bill",
        "right in the childhood",
        "braveheart",
        "keep calm and carry on black",
        "scumbag minecraft",
        "they hated jesus meme",
        "for dummies",
        "george bush",
        "bart simpson - chalkboard",
        "business cat",
        "surprised koala",
        "two buttons 1 blue",
        "urinal guy more text room",
        "aw yeah rage face",
        "bazooka squirrel",
        "inspirational quote",
        "joe biden",
        "brian williams was there 2",
        "fat asian kid",
        "onde",
        "why is the fbi here",
        "eye of sauron",
        "sonic says",
        "jim lehrer the man",
        "bart simpson peeking",
        "cmon do something",
        "y u no",
        "lame pun coon",
        "cnn breaking news template",
        "shia labeouf just do it",
        "you the real mvp 2",
        "tech impaired duck",
        "bonjour",
        "hardworking guy",
        "keep calm and carry on aqua",
        "pepe the frog",
        "hey internet",
        "captain america so you",
        "blank pokemon card",
        "liam neeson taken",
        "big book small book",
        "internet explorer",
        "computer guy",
        "comic book guy",
        "scumbag brain",
        "blank comic panel 2x1",
        "no take only throw",
        "blank",
        "farquaad pointing",
        "sigmund freud",
        "triggerpaul",
        "extra-hell",
        "blank for president",
        "mario hammer smash",
        "money man",
        "there is 1 imposter among us",
        "the most interesting cat in the world",
        "wanted poster",
        "computer horse",
        "jersey santa",
        "black background",
        "silent movie card",
        "2nd term obama",
        "foul bachelor frog",
        "i know that feel bro",
        "scared cat",
        "transparent",
        "first world problems cat",
        "meme man",
        "jeopardy blank",
        "unhappy baby",
        "table flip guy",
        "you made thisi made this",
        "mexican word of the day",
        "socially awkward awesome penguin",
        "gravestone",
        "challenge accepted rage face",
        "blank transparent square",
        "fist pump baby",
        "blank black",
        "awkward moment sealion",
        "zuckerberg",
        "original stoner dog",
        "you underestimate my power",
        "hohoho",
        "angry birds pig",
        "keep calm and carry on purple",
        "arrogant rich man",
        "no soup",
        "facepalm bear",
        "evil plotting raccoon",
        "marked safe from",
        "dolph ziggler sells",
        "alarm clock",
        "pepe silvia",
        "nobody absolutely no one",
        "honey its time to x",
        "hold my beer",
        "who would win",
        "now thats what i call",
        "gadsden flag",
        "hipster ariel",
        "keep calm and carry on red",
        "hulk",
        "confucius says",
        "free",
        "grumpy toad",
        "laughing villains",
        "blank blue background",
        "stay alert",
        "buddy the elf",
        "brian williams was there",
        "wtf",
        "pokemon appears",
        "cute cat",
        "forever alone",
        "team rocket",
        "rod serling twilight zone",
        "heavy breathing cat",
        "warning sign",
        "white background",
        "blank white template",
        "blank starter pack",
        "metal jesus",
        "blank t-shirt",
        "iceberg",
        "happy star congratulations",
        "short satisfaction vs truth",
        "serious xzibit",
        "multi doge",
        "malicious advice mallard",
        "mr t pity the fool",
        "banned from roblox",
        "among us not the imposter",
        "cereal guy",
        "homophobic seal",
        "neil degrasse tyson",
        "nooo haha go brrr",
        "you dont say",
        "austin powers honestly",
        "gotta go cat",
        "impossibru guy original",
        "blank comic panel 1x2",
        "troll face",
        "milk carton",
        "yeet the child",
        "blank yellow sign",
        "take a seat cat",
    ]
)

REPLICATED_TEMPLATES_GROUPED = [
    [
        "no - yes",
        "drake blank",
        "drake meme",
        "drake hotline approves",
        "drake hotline bling",
        "drake noyes",
    ],
    [
        "baby yoda die trash",
        "surprised baby yoda",
        "crying baby yoda",
        "sad baby yoda",
        "baby yoda tea",
        "baby yoda",
    ],
    ["mr incredible mad", "math is math"],
    ["nemo birds", "nemo seagulls mine"],
    [
        "anime butterfly meme",
        "butterfly man",
        "is this a bird",
        "is this butterfly",
        "is this a pigeon",
        "is this a blank",
        "is this a pigeon",
    ],
    ["wait its all", "always has been"],
    ["admiral ackbar relationship expert", "its a trap", "admiral ackbar"],
    ["afraid to ask andy closeup", "afraid to ask andy"],
    ["all my homies hate", "all my homies love", "all my homies"],
    ["aliens guy", "ancient aliens"],
    [
        "woman pointing at cat",
        "woman yelling at cat",
        "woman yelling at cat",
        "woman yelling at cat",
        "smudge the cat",
        "white cat table",
        "angry lady cat",
        "cat at table",
        "lady screams at cat",
    ],
    ["astronaut meme always has been template", "always has been"],
    ["batman slaps trump", "batman slapping robin"],
    ["bernie sanders mittens", "bernie sitting", "bernie mittens"],
    [
        "i am once again asking",
        "bernie i am once again asking for your support",
        "bernie sanders once again asking",
    ],
    [
        "im about to end this mans whole career",
        "im about to end this mans whole career",
    ],
    ["jordan peele sweating", "sweating bullets"],
    ["hide the pain harold", "hide the pain harold"],
    ["me and the boys", "me and the boys"],
    ["monsters inc", "sully wazowski"],
    [
        "thinking black guy",
        "smart black guy",
        "you cant if you dont",
        "you cant if you dont",
        "smart guy",
        "guy tapping head",
        "roll safe think about it",
        "black guy pointing at head",
        "black guy head tap",
        "you cant if you dont",
        "eddie murphy thinking",
    ],
    ["bender", "blackjack and hookers"],
    ["brace yourselves x is coming", "winter is coming", "brace yourself"],
    [
        "robotnik pressing red button",
        "both buttons pressed",
        "press button hard choice",
        "two buttons",
    ],
    ["woody and buzz lightyear everywhere widescreen", "x x everywhere"],
    ["braveheart freedom", "braveheart"],
    ["black guy stopping", "bro not cool"],
    ["buddy the elf excited", "buddy the elf"],
    [
        "buff doge vs cheems",
        "big dog small dog",
        "strong dog vs weak dog",
        "buff doge vs crying cheems",
    ],
    [
        "kermit sipping tea",
        "but thats none of my business neutral",
        "but thats none of my business",
    ],
    ["car salesman slaps hood", "car salesman slaps roof of car"],
    ["car drift meme", "car turning", "left exit 12 off ramp"],
    [
        "he will never get a girlfriend",
        "cereal guy spitting",
        "cereal guys daddy",
        "cereal guy",
    ],
    ["change my mind crowder", "prove me wrong", "change my mind"],
    [
        "charlie conspiracy always sunny in philidelphia",
        "trying to explain",
        "charlie day",
    ],
    [
        "chuck norris approves",
        "chuck norris finger",
        "chuck norris flex",
        "chuck norris guns",
        "chuck norris laughing",
        "chuck norris phone",
        "chuck norris with guns",
        "chuck norris",
    ],
    ["cmon do something", "cmon do something"],
    ["comic book guy worst ever", "comic book guy"],
    ["bugs bunny communist", "our", "communist bugs bunny"],
    ["computer guy facepalm", "computer horse", "computer guy"],
    ["adive yoda", "star wars yoda"],
    ["thanos perfectly balanced as all things should be", "thanos perfectly balanced"],
    [
        "crack",
        "crack head",
        "tyrone biggums the addict",
        "dave chappelle yall got any more of crackhead",
        "yall got any more of them",
        "dave chappelle",
        "yall got any more of that",
        "yall got any more of",
    ],
    ["crying cat", "crying cat"],
    ["oh yeah! oh no", "black guy happy sad", "disappointed black guy"],
    ["peter parker cry", "crying peter parker"],
    ["burning house girl", "disaster girl"],
    ["distracted boyfriend", "distracted boyfriend"],
    ["wait thats illegal", "wait thats illegal"],
    ["ralph in danger", "im in danger + blank place above", "chuckles im in danger"],
    ["archer", "do you want ants archer"],
    ["virgin and chad", "virgin vs chad"],
    ["doge 2", "doge"],
    [
        "make america great again",
        "trump bill signing",
        "trump lawn mower",
        "donal trump birthday",
        "donald trump approves",
        "donald trump birthday",
        "donald trump",
    ],
    [
        "dr evil air quotes",
        "dr evil laser",
        "dr evil pinky",
        "dr evil quotes",
        "dr evil",
    ],
    ["dwight schrute 2", "dwight schrute"],
    ["black white arms", "predator handshake", "epic handshake"],
    [
        "galaxy brain 3 brains",
        "expanding brain 3 panels",
        "expanding brain 5 panel",
        "expanding brain",
    ],
    [
        "fancy winnie the pooh meme",
        "tuxedo winnie the pooh 3 panel",
        "tuxedo winnie the pooh 4 panel",
        "tuxedo winnie the pooh",
        "bestbetter blurst",
        "fancy pooh",
    ],
    ["black scientist finally xium", "finally"],
    ["for dummies book", "for dummies"],
    ["forever alone christmas", "forever alone happy", "forever alone"],
    ["blue futurama fry", "not sure if- fry", "futurama fry"],
    ["gangnam style psy", "psy horse dance", "gangnam style2", "gangnam style"],
    ["gru diabolical plan fail", "grus plan"],
    ["aj styles & undertaker", "randy orton undertaker", "undertaker"],
    ["zombie bad luck brian", "bad luck brian"],
    [
        "grumpy cats father",
        "grumpy cat bed",
        "grumpy cat birthday",
        "grumpy cat christmas",
        "grumpy cat does not believe",
        "grumpy cat halloween",
        "grumpy cat happy",
        "grumpy cat mistletoe",
        "grumpy cat not amused",
        "grumpy cat reverse",
        "grumpy cat sky",
        "grumpy cat star wars",
        "grumpy cat table",
        "grumpy cat top hat",
        "grumpy cat",
    ],
    ["homer simpson in bush - large", "homer bush", "homer disappears into bush"],
    ["here we go again", "ah shit here we go again"],
    [
        "hes probably thinking about girls",
        "thinking of other girls",
        "i bet hes thinking about other women",
    ],
    ["joe biden 2020", "sad joe biden", "joe biden worries", "joe biden"],
    ["krusty krab vs chum bucket blank", "krusty krab vs chum bucket"],
    ["leonardo dicaprio django laugh", "laughing leo"],
    ["madea", "madea with gun"],
    ["marvel civil war 1", "marvel civil war 2", "marvel civil war"],
    ["side eye teddy", "monkey puppet", "monkey looking away"],
    ["hold fart", "when you havent told anybody", "neck vein guy", "straining kid"],
    ["nick young", "black guy confused"],
    ["npc meme", "npc"],
    ["phoebe teaching joey in friends", "phoebe joey"],
    ["persian cat room guardian single", "persian cat room guardian"],
    ["rick and carl 3", "rick and carl long", "rick and carl longer", "rick and carl"],
    ["samuel jackson glance", "samuel l jackson", "samuel l jackson"],
    [
        "spider man double",
        "spider man triple",
        "spiderman camera",
        "spiderman computer desk",
        "spiderman hospital",
        "spiderman mirror",
        "spiderman pointing at spiderman",
        "spiderman presentation",
        "sexy railroad spiderman",
        "spiderman",
    ],
    ["dog on fire", "this is fine"],
    ["what do we want 3", "what do we want", "x all the y"],
    ["sponge finna commit muder", "spongebob strong"],
    ["spongebob stupid", "mocking spongebob"],
    ["soldier protecting sleeping child", "the silent protector"],
    ["spongebob rainbow", "imagination spongebob"],
    ["empty stonks", "stonks without stonks", "stonks"],
    ["the office congratulations", "the office handshake"],
    ["this is fine blank", "this is fine"],
    ["timmys turner dad", "this is where id put my trophy if i had one"],
    ["triggered liberal", "triggered feminist"],
    [
        "vince mcmahon reaction wglowing eyes",
        "mr mcmahon reaction",
        "vince mcmahon reaction",
        "vince mcmahon reaction",
        "mcmahon",
        "vince mcmahon",
        "vince mcmahon",
    ],
    ["waiting skeleton", "waiting skeleton"],
    ["we dont do that here", "we dont do that here"],
    ["captain phillips - im the captain now", "im the captain now", "look at me"],
    ["oprah you get a car everybody gets a car", "oprah you get a"],
    ["drowning kid in the pool", "mother ignoring kid drowning in a pool"],
    ["oh no anyway", "oh no! anyway"],
    ["licking lips", "black guy hiding behind tree", "anthony adams rubbing hands"],
    [
        "alien meeting suggestion",
        "emergency meeting among us",
        "among us meeting",
        "boardroom meeting suggestion",
    ],
    ["matrix morpheus", "what if i told you"],
    ["party loner", "they dont know"],
]
COMPRESSED_NAMES = set(group[-1] for group in REPLICATED_TEMPLATES_GROUPED)
ALT_NAMES = set(item for group in REPLICATED_TEMPLATES_GROUPED for item in group[:-1])

SEEN_MEMES = set(
    [
        "blackjack and hookers",
        "pimples zero!",
        "i find your lack of faith disturbing",
        "omg karen",
        "triggered feminist",
        "ill just wait here",
        "waiting skeleton",
        "this morgan freeman",
        "genie rules meme",
        "sad linus",
        "dad joke dog",
        "i believe in supremacy",
        "perfection",
        "the boiler room of hell",
        "expanding brain",
        "anime wall punch",
        "alright gentlemen we need a new idea",
        "homer simpsons back fat",
        "god",
        "persian cat room guardian",
        "stop it patrick! youre scaring him!",
        "confused gandalf",
        "confused unga bunga",
        "we ride at dawn bitches",
        "pentagon hexagon octagon",
        "you have no power here",
        "snape",
        "always has been",
        "ptsd clarinet boy",
        "trojan horse",
        "is for me",
        "angry chef gordon ramsay",
        "ptsd chihuahua",
        "overly attached girlfriend",
        "scumbag steve",
        "evil kermit",
        "this little manuever is gonna cost us 51 years",
        "phoebe joey",
        "disappointed black guy",
        "cool cat stroll",
        "calculating meme",
        "here it come meme",
        "mother ignoring kid drowning in a pool",
        "well boys we did it blank is no more",
        "but thats none of my business",
        "oh boy here i go killing again",
        "assassination chain",
        "borat",
        "black guy disappearing",
        "my time has come",
        "1990s first world problems",
        "suprised patrick",
        "the cooler daniel",
        "the floor is",
        "vince mcmahon",
        "srgrafo dude wtf",
        "man giving sword to larger man",
        "ppap",
        "listen here you little shit bird",
        "mugatu so hot right now",
        "meg family guy better than me",
        "types of headaches meme",
        "i should buy a boat cat",
        "its free real estate",
        "dabbing dude",
        "nemo seagulls mine",
        "whatcha got there",
        "what if i told you",
        "x x everywhere",
        "no i dont think i will",
        "billys fbi agent",
        "turn up the volume",
        "so anyway i started blasting",
        "friendship ended",
        "note passing",
        "wait thats illegal",
        "what gives people feelings of power",
        "angry turkish man playing cards meme",
        "inhaling seagull",
        "nuclear explosion",
        "floating boy chasing running boy",
        "thanos what did it cost",
        "bro not cool",
        "finally",
        "scared kid",
        "high five drown",
        "one does not simply",
        "grant gustin over grave",
        "finding neverland",
        "charlie day",
        "steve harvey laughing serious",
        "oprah you get a",
        "i sleep real shit",
        "squidward window",
        "hey you going to sleep",
        "grumpy cat",
        "skeleton waiting",
        "dw sign wont stop me because i cant read",
        "admiral ackbar",
        "well yes but actually no",
        "blankie the shocked dog",
        "stonks helth",
        "samuel l jackson",
        "i dont want to play with you anymore",
        "do you want ants archer",
        "sparta leonidas",
        "domino effect",
        "spiderman peter parker",
        "laughing leo",
        "spongebob hype stand",
        "girl running",
        "soldier jump spetznaz",
        "car salesman slaps roof of car",
        "minor mistake marvin",
        "i am inevitable",
        "squidward",
        "spongegar",
        "mr bean copying",
        "they dont know",
        "pointing mirror guy",
        "joey from friends",
        "sleepy donald duck in bed",
        "blank nut button",
        "kermit window",
        "thanos infinity stones",
        "but it was me dio",
        "gordon ramsay some good food",
        "its been 84 years",
        "bernie mittens",
        "arthur fist",
        "homer disappears into bush",
        "spongebob diapers meme",
        "angela scared dwight",
        "guy pouring olive oil on the salad",
        "batman slapping robin",
        "every day i wake up",
        "kevin hart",
        "boy and girl texting",
        "bad pun dog",
        "for christmas i want a dragon",
        "you guys are getting paid template",
        "thor is he though",
        "and thats a fact",
        "stonks",
        "straining kid",
        "dj khaled suffering from success meme",
        "buff doge vs crying cheems",
        "marvel civil war",
        "finally! a worthy opponent!",
        "bro explaining",
        "how tough are you",
        "surprised pikachu",
        "dont you squidward",
        "baby yoda",
        "kombucha girl",
        "pink guy vs bane",
        "nobody is born cool",
        "la noire press x to doubt",
        "drake noyes",
        "sure grandma lets get you to bed",
        "dont make me tap the sign",
        "brace yourself",
        "young thug and lil durk troubleshooting",
        "mr krabs blur meme",
        "aaaaand its gone",
        "you wouldnt get it",
        "knights of the round table",
        "laughing men in suits",
        "panik kalm panik",
        "yugioh card draw",
        "the future world if",
        "sad keanu",
        "gru gun",
        "we dont do that here",
        "the office bankruptcy",
        "the mandalorian",
        "who killed hannibal",
        "distracted boyfriend",
        "yall got any more of",
        "shrek for five minutes",
        "first time",
        "blank kermit waiting",
        "ew i stepped in shit",
        "patrick not my wallet",
        "metronome",
        "afraid to ask andy",
        "get in loser",
        "this is where id put my trophy if i had one",
        "donald trump",
        "tom cruise laugh",
        "ermahgerd berks",
        "gollum",
        "its finally over",
        "spiderman laugh",
        "life is good but it can be better",
        "dr evil",
        "impostor of the vent",
        "npc",
        "what if you wanted to go to heaven",
        "modern problems",
        "jim halpert explains",
        "call an ambulance but not for me",
        "this is brilliant but i like this",
        "success kid",
        "i do one push-up",
        "bike fall",
        "change my mind",
        "jack sparrow you have heard of me",
        "daring today arent we squidward",
        "patrick smart dumb",
        "why are you gay",
        "let me in",
        "scooby doo mask reveal",
        "cross eyed spongebob",
        "third world success kid",
        "gang bang",
        "grandma finds the internet",
        "buddy christ",
        "theyre the same picture",
        "i wish all the x a very pleasant evening",
        "left exit 12 off ramp",
        "star wars no",
        "black guy confused",
        "laughing wolf",
        "thanos perfectly balanced",
        "adios",
        "boardroom meeting suggestion",
        "sarcastically surprised kirk",
        "crying michael jordan",
        "philosoraptor",
        "yo dawg heard you",
        "dinkleberg",
        "am i a joke to you",
        "sweating bullets",
        "chuck norris",
        "monkey looking away",
        "kill yourself guy",
        "if those kids could read theyd be very upset",
        "skinner out of touch",
        "thats a paddlin",
        "crying peter parker",
        "salt bae",
        "pretending to be happy hiding crying behind a mask",
        "metal jesus",
        "a train hitting a school bus",
        "im the dumbest man alive",
        "all my homies",
        "guy with sand in the hands of despair",
        "math is math",
        "imagination spongebob",
        "weak vs strong spongebob",
        "dog vs werewolf",
        "doge",
        "obi wan kenobi",
        "whats my purpose - butter robot",
        "trumpet boy",
        "willy wonka blank",
        "elmo fire",
        "ill have you know spongebob",
        "futurama fry",
        "crying cat",
        "joe exotic financially recover",
        "karate kyle",
        "aint nobody got time for that",
        "krusty krab vs chum bucket",
        "thomas had never seen such bullshit before",
        "who wants to be a millionaire",
        "so you have chosen death",
        "two buttons",
        "running away balloon",
        "leonardo dicaprio pointing",
        "bronze medal",
        "spiderman glasses",
        "moe throws barney",
        "undertaker",
        "me and the boys",
        "grim reaper knocking door",
        "captain picard facepalm",
        "you cant defeat me",
        "im about to end this mans whole career",
        "chocolate spongebob",
        "lumbergh",
        "shut up and take my money fry",
        "anthony adams rubbing hands",
        "spongebob strong",
        "all right then keep your secrets",
        "epic handshake",
        "guy holding cardboard sign",
        "trump interview",
        "inception",
        "relief",
        "jojos walk",
        "it aint much but its honest work",
        "disaster girl",
        "uno draw 25 cards",
        "angry pakistani fan",
        "this is worthless",
        "i am speed",
        "skinner pathetic",
        "not stonks",
        "mel gibson and jesus christ",
        "squidward chair",
        "red pill blue pill",
        "excuse me wtf blank template",
        "trying to calculate how much sleep i can get",
        "see nobody cares",
        "yeah this is big brain time",
        "eddie murphy thinking",
        "highdrunk guy",
        "communist bugs bunny",
        "i want you",
        "spongebob ight imma head out",
        "advice dog",
        "trust nobody not even yourself",
        "amateurs",
        "oh no! anyway",
        "where banana",
        "you guys always act like youre better than me",
        "twisted tea",
        "i fear no man",
        "you guys are getting paid",
        "clown applying makeup",
        "average blank fan vs average blank enjoyer",
        "spongebob burning paper",
        "idiot skull",
        "back in my day",
        "simpsons so far",
        "fallout hold up",
        "lisa simpson coffee that x shit",
        "unsettled tom",
        "spongebob money",
        "sue sylvester",
        "joaquin phoenix joker car",
        "i am the senate",
        "ancient aliens",
        "rainbow six - fuze the hostage",
        "open the gate a little",
        "ah shit here we go again",
        "bernie sanders once again asking",
        "chuckles im in danger",
        "visible confusion",
        "good fellas hilarious",
        "is this a pigeon",
        "me explaining to my mom",
        "the silent protector",
        "put it somewhere else patrick",
        "when x just right",
        "flex tape",
        "bernie sanders reaction nuked",
        "look at me",
        "dwight schrute",
        "pepperidge farm remembers",
        "steve buscemi fellow kids",
        "captain america elevator",
        "money money",
        "say the line bart! simpsons",
        "leonardo dicaprio cheers",
        "this is fine",
        "and just like that",
        "spongebob waiting",
        "and everybody loses their minds",
        "spiderman",
        "the loudest sounds on earth",
        "wolverine remember",
        "simba shadowy place",
        "my heart blank",
        "bad luck brian",
        "lisa simpsons presentation",
        "hide the pain harold",
        "sully wazowski",
        "the office handshake",
        "star wars yoda",
        "third world skeptical kid",
        "wolf of wallstreet",
        "mario bros views",
        "say what again",
        "aww did someone get addicted to crack",
        "peter griffin news",
        "why cant you just be normal",
        "the most interesting man in the world",
        "lady screams at cat",
        "grus plan",
        "prisoners blank",
        "elmo cocaine",
        "madea with gun",
        "feels good man",
        "annoyed bird",
        "jack sparrow being chased",
        "second breakfast",
        "the scroll of truth",
        "unhelpful high school teacher",
        "jason momoa henry cavill meme",
        "fancy pooh",
        "i bet hes thinking about other women",
        "mike wazowski trying to explain",
        "elmo nuclear explosion",
        "allow us to introduce ourselves",
        "the rock driving",
        "bird box",
        "jack nicholson the shining snow",
        "kim jong un sad",
        "the what",
        "satisfied seal",
        "black girl wat",
        "x all the y",
        "too damn high",
        "thanos impossible",
        "rick and morty-extra steps",
        "and i took that personally",
        "confused screaming",
        "rick and carl",
        "mocking spongebob",
        "maybe i am a monster",
        "mr bean waiting",
        "anime girl hiding from terminator",
        "you know im something of a scientist myself",
        "consuela",
        "first world problems",
        "virgin vs chad",
    ]
)
MEMES_TO_USE: List[str] = list(SEEN_MEMES - ALT_NAMES - DONT_USE_TEMPLATES)
