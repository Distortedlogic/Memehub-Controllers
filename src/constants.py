from pathlib import Path
from typing import Dict, List, cast

import arrow
from arrow.arrow import Arrow

LOAD_MEME_CLF_VERSION = "0.4.4"
LOAD_STONK_VERSION = "0.0"
MEME_CLF_VERSION = "0.4.4"
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
LOAD_STATIC_PATH=MODELS_REPO + "market/{}/"

Path(MODELS_REPO).mkdir(parents=True, exist_ok=True)  # type: ignore
Path(NOT_MEME_REPO).mkdir(parents=True, exist_ok=True)  # type: ignore
Path(NOT_TEMPLATE_REPO).mkdir(parents=True, exist_ok=True)  # type: ignore
Path(MEMES_REPO).mkdir(parents=True, exist_ok=True)  # type: ignore
Path(INCORRECT_REPO).mkdir(parents=True, exist_ok=True)  # type: ignore
Path(BLANKS_REPO).mkdir(parents=True, exist_ok=True)  # type: ignore

for folder in ["reg", "jit", "cp"]:
    Path(MEME_CLF_REPO.format(folder)).mkdir(parents=True, exist_ok=True)
    Path(STONK_REPO.format(folder)).mkdir(parents=True, exist_ok=True)
    Path(NOT_A_MEME_MODEL_REPO.format(folder)).mkdir(parents=True, exist_ok=True)

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
        "hardworking guy",
        "arrogant rich man",
        "meme man",
        "computer guy",
        "farquaad pointing",
        "captain america so you",
        "buddy the elf",
        "smiling cat",
        "joe biden",
        "angry baby",
        "business cat",
        "mexican word of the day",
        "they hated jesus meme",
        "original stoner dog",
        "mr t pity the fool",
        "you the real mvp 2",
        "mario hammer smash",
        "scared cat",
        "yeet the child",
        "the most interesting cat in the world",
        "pepe the frog",
        "cnn breaking news template",
        "fist pump baby",
        "first world problems cat",
        "confucius says",
        "rod serling twilight zone",
        "braveheart",
        "you underestimate my power",
        "cute cat",
        "bazooka squirrel",
        "unhappy baby",
        "internet explorer",
        "take a seat cat",
        "foul bachelor frog",
        "jersey santa",
        "zuckerberg",
        "happy star congratulations",
        "metal jesus",
        "slenderman",
        "socially awkward awesome penguin",
        "wtf",
        "malicious advice mallard",
        "short satisfaction vs truth",
        "gotta go cat",
        "speechless colbert face",
        "scumbag brain",
        "why is the fbi here",
        "if you know what i mean bean",
        "mr bean",
        "wants to know your location",
        "blank pokemon card",
        "pokemon appears",
        "team rocket",
        "aw yeah rage face",
        "cereal guy",
        "extra-hell",
        "big book small book",
        "cmon do something",
        "valentines day card meme",
        "sonic says",
        "facepalm bear",
        "i know that feel bro",
        "laughing villains",
        "among us not the imposter",
        "blank colored background",
        "gangnam style",
        "money man",
        "stay alert",
        "for dummies",
        "blank comic panel 1x2",
        "free",
        "banned from roblox",
        "jim lehrer the man",
        "white background",
        "angry birds pig",
        "hohoho",
        "blank yellow sign",
        "nobody absolutely no one",
        "comic book guy",
        "shia labeouf just do it",
        "dancing trollmom",
        "computer horse",
        "challenge accepted rage face",
        "lame pun coon",
        "silent movie card",
        "blank white template",
        "warning sign",
        "gadsden flag",
        "blank blue background",
        "black background",
        "keep calm and carry on black",
        "scumbag minecraft",
        "bart simpson - chalkboard",
        "impossibru guy original",
        "fat asian kid",
        "does your dog bite",
        "blank starter pack",
        "honey its time to x",
        "blank black",
        "keep calm and carry on aqua",
        "bart simpson peeking",
        "eye of sauron",
        "who would win",
        "now thats what i call",
        "evil plotting raccoon",
        "triggerpaul",
        "tui",
        "be like bill",
        "pepe silvia",
        "marked safe from",
        "two buttons 1 blue",
        "hipster ariel",
        "right in the childhood",
        "keep calm and carry on red",
        "george bush",
        "table flip guy",
        "transparent",
        "you made thisi made this",
        "milk carton",
        "forever alone",
        "blank comic panel 2x2",
        "keep calm and carry on purple",
        "hulk",
        "baby crying",
        "y u no",
        "inspirational quote",
        "grumpy toad",
        "2nd term obama",
        "austin powers honestly",
        "liam neeson taken",
        "brian williams was there 3",
        "blank for president",
        "heavy breathing cat",
        "sigmund freud",
        "no soup",
        "no soup for you",
        "nooo haha go brrr",
        "bonjour",
        "urinal guy more text room",
        "wanted poster",
        "hold my beer",
        "blank",
        "brian williams was there 2",
        "gasp rage face",
        "homophobic seal",
        "blank transparent square",
        "tech impaired duck",
        "troll face",
        "reverse kalm panik",
        "multi doge",
        "what my friends think i do",
        "no take only throw",
        "neil degrasse tyson",
        "blank t-shirt",
        "there is 1 imposter among us",
        "mom can we have",
        "you dont say",
        "surprised koala",
        "iceberg",
        "jeopardy blank",
        "brian williams was there",
        "hey internet",
        "dumpster fire",
        "dolph ziggler sells",
        "noooo you cant just",
        "alarm clock",
        "onde",
        "gravestone",
        "blank comic panel 2x1",
        "serious xzibit",
        "awkward moment sealion",
        "permission bane",
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
        "1990s first world problems",
        "a train hitting a school bus",
        "aaaaand its gone",
        "adios",
        "admiral ackbar",
        "advice dog",
        "afraid to ask andy",
        "ah shit here we go again",
        "aint nobody got time for that",
        "all my homies",
        "all right then keep your secrets",
        "allow us to introduce ourselves",
        "alright gentlemen we need a new idea",
        "always has been",
        "am i a joke to you",
        "amateurs",
        "ancient aliens",
        "and everybody loses their minds",
        "and i took that personally",
        "and just like that",
        "and thats a fact",
        "angela scared dwight",
        "angry chef gordon ramsay",
        "angry pakistani fan",
        "angry turkish man playing cards meme",
        "anime girl hiding from terminator",
        "anime wall punch",
        "annoyed bird",
        "anthony adams rubbing hands",
        "arthur fist",
        "assassination chain",
        "average blank fan vs average blank enjoyer",
        "aww did someone get addicted to crack",
        "baby yoda",
        "back in my day",
        "bad luck brian",
        "bad pun dog",
        "batman slapping robin",
        "bernie mittens",
        "bernie sanders once again asking",
        "bernie sanders reaction nuked",
        "bike fall",
        "billys fbi agent",
        "bird box",
        "black girl wat",
        "black guy confused",
        "black guy disappearing",
        "blackjack and hookers",
        "blank kermit waiting",
        "blank nut button",
        "blankie the shocked dog",
        "boardroom meeting suggestion",
        "borat",
        "boy and girl texting",
        "brace yourself",
        "bro explaining",
        "bro not cool",
        "bronze medal",
        "buddy christ",
        "buff doge vs crying cheems",
        "but it was me dio",
        "but thats none of my business",
        "calculating meme",
        "call an ambulance but not for me",
        "captain america elevator",
        "captain picard facepalm",
        "car salesman slaps roof of car",
        "change my mind",
        "charlie day",
        "chocolate spongebob",
        "chuck norris",
        "chuckles im in danger",
        "clown applying makeup",
        "communist bugs bunny",
        "confused gandalf",
        "confused screaming",
        "confused unga bunga",
        "consuela",
        "cool cat stroll",
        "cross eyed spongebob",
        "crying cat",
        "crying michael jordan",
        "crying peter parker",
        "dabbing dude",
        "dad joke dog",
        "daring today arent we squidward",
        "dinkleberg",
        "disappointed black guy",
        "disaster girl",
        "distracted boyfriend",
        "dj khaled suffering from success meme",
        "do you want ants archer",
        "dog vs werewolf",
        "doge",
        "domino effect",
        "donald trump",
        "dont make me tap the sign",
        "dont you squidward",
        "dr evil",
        "drake noyes",
        "dw sign wont stop me because i cant read",
        "dwight schrute",
        "eddie murphy thinking",
        "elmo cocaine",
        "elmo fire",
        "elmo nuclear explosion",
        "epic handshake",
        "ermahgerd berks",
        "every day i wake up",
        "evil kermit",
        "ew i stepped in shit",
        "excuse me wtf blank template",
        "expanding brain",
        "fallout hold up",
        "fancy pooh",
        "feels good man",
        "finally",
        "finally! a worthy opponent!",
        "finding neverland",
        "first time",
        "first world problems",
        "flex tape",
        "floating boy chasing running boy",
        "for christmas i want a dragon",
        "friendship ended",
        "futurama fry",
        "gang bang",
        "genie rules meme",
        "get in loser",
        "girl running",
        "god",
        "gollum",
        "good fellas hilarious",
        "gordon ramsay some good food",
        "grandma finds the internet",
        "grant gustin over grave",
        "grim reaper knocking door",
        "gru gun",
        "grumpy cat",
        "grus plan",
        "guy holding cardboard sign",
        "guy pouring olive oil on the salad",
        "guy with sand in the hands of despair",
        "here it come meme",
        "hey you going to sleep",
        "hide the pain harold",
        "high five drown",
        "highdrunk guy",
        "homer disappears into bush",
        "homer simpsons back fat",
        "how tough are you",
        "i am inevitable",
        "i am speed",
        "i am the senate",
        "i believe in supremacy",
        "i bet hes thinking about other women",
        "i do one push-up",
        "i dont want to play with you anymore",
        "i fear no man",
        "i find your lack of faith disturbing",
        "i should buy a boat cat",
        "i sleep real shit",
        "i want you",
        "i wish all the x a very pleasant evening",
        "idiot skull",
        "if those kids could read theyd be very upset",
        "ill have you know spongebob",
        "ill just wait here",
        "im about to end this mans whole career",
        "im the dumbest man alive",
        "imagination spongebob",
        "impostor of the vent",
        "inception",
        "inhaling seagull",
        "is for me",
        "is this a pigeon",
        "it aint much but its honest work",
        "its been 84 years",
        "its finally over",
        "its free real estate",
        "jack nicholson the shining snow",
        "jack sparrow being chased",
        "jack sparrow you have heard of me",
        "jason momoa henry cavill meme",
        "jim halpert explains",
        "joaquin phoenix joker car",
        "joe exotic financially recover",
        "joey from friends",
        "jojos walk",
        "jordan peele sweating",
        "karate kyle",
        "kermit window",
        "kevin hart",
        "kill yourself guy",
        "kim jong un sad",
        "knights of the round table",
        "kombucha girl",
        "krusty krab vs chum bucket",
        "la noire press x to doubt",
        "lady screams at cat",
        "laughing leo",
        "laughing men in suits",
        "laughing wolf",
        "left exit 12 off ramp",
        "leonardo dicaprio cheers",
        "leonardo dicaprio pointing",
        "let me in",
        "life is good but it can be better",
        "lisa simpson coffee that x shit",
        "lisa simpsons presentation",
        "listen here you little shit bird",
        "look at me",
        "lumbergh",
        "madea with gun",
        "man giving sword to larger man",
        "mario bros views",
        "marvel civil war",
        "math is math",
        "maybe i am a monster",
        "me and the boys",
        "me explaining to my mom",
        "meg family guy better than me",
        "mel gibson and jesus christ",
        "metal jesus",
        "metronome",
        "mike wazowski trying to explain",
        "minor mistake marvin",
        "mocking spongebob",
        "modern problems",
        "moe throws barney",
        "money money",
        "monkey looking away",
        "mother ignoring kid drowning in a pool",
        "mr bean copying",
        "mr bean waiting",
        "mr krabs blur meme",
        "mugatu so hot right now",
        "my heart blank",
        "my time has come",
        "nemo seagulls mine",
        "no i dont think i will",
        "nobody is born cool",
        "not stonks",
        "note passing",
        "npc",
        "nuclear explosion",
        "obi wan kenobi",
        "oh boy here i go killing again",
        "oh no! anyway",
        "omg karen",
        "one does not simply",
        "open the gate a little",
        "oprah you get a",
        "overly attached girlfriend",
        "panik kalm panik",
        "patrick not my wallet",
        "patrick smart dumb",
        "pentagon hexagon octagon",
        "pepperidge farm remembers",
        "perfection",
        "persian cat room guardian",
        "peter griffin news",
        "philosoraptor",
        "phoebe joey",
        "pimples zero!",
        "pink guy vs bane",
        "pointing mirror guy",
        "ppap",
        "pretending to be happy hiding crying behind a mask",
        "prisoners blank",
        "ptsd chihuahua",
        "ptsd clarinet boy",
        "put it somewhere else patrick",
        "rainbow six - fuze the hostage",
        "red pill blue pill",
        "relief",
        "rick and carl",
        "rick and morty-extra steps",
        "running away balloon",
        "sad keanu",
        "salt bae",
        "samuel l jackson",
        "sarcastically surprised kirk",
        "satisfied seal",
        "say the line bart! simpsons",
        "say what again",
        "scared kid",
        "scooby doo mask reveal",
        "scumbag steve",
        "second breakfast",
        "see nobody cares",
        "shrek for five minutes",
        "shut up and take my money fry",
        "simba shadowy place",
        "simpsons so far",
        "skeleton waiting",
        "skinner out of touch",
        "skinner pathetic",
        "sleepy donald duck in bed",
        "snape",
        "so anyway i started blasting",
        "so you have chosen death",
        "soldier jump spetznaz",
        "soldier protecting sleeping child",
        "sparta leonidas",
        "spiderman",
        "spiderman glasses",
        "spiderman laugh",
        "spiderman peter parker",
        "spongebob burning paper",
        "spongebob diapers meme",
        "spongebob hype stand",
        "spongebob ight imma head out",
        "spongebob money",
        "spongebob rainbow",
        "spongebob strong",
        "spongebob waiting",
        "spongegar",
        "squidward",
        "squidward chair",
        "squidward window",
        "srgrafo dude wtf",
        "star wars no",
        "star wars yoda",
        "steve buscemi fellow kids",
        "steve harvey laughing serious",
        "stonks",
        "stonks helth",
        "stop it patrick! youre scaring him!",
        "straining kid",
        "success kid",
        "sue sylvester",
        "sully wazowski",
        "suprised patrick",
        "sure grandma lets get you to bed",
        "surprised pikachu",
        "sweating bullets",
        "thanos impossible",
        "thanos infinity stones",
        "thanos perfectly balanced",
        "thanos what did it cost",
        "thats a paddlin",
        "the boiler room of hell",
        "the cooler daniel",
        "the floor is",
        "the future world if",
        "the loudest sounds on earth",
        "the mandalorian",
        "the most interesting man in the world",
        "the office bankruptcy",
        "the office handshake",
        "the rock driving",
        "the scroll of truth",
        "the silent protector",
        "the what",
        "they dont know",
        "theyre the same picture",
        "third world skeptical kid",
        "third world success kid",
        "this is brilliant but i like this",
        "this is fine",
        "this is where id put my trophy if i had one",
        "this is worthless",
        "this little manuever is gonna cost us 51 years",
        "this morgan freeman",
        "thomas had never seen such bullshit before",
        "thor is he though",
        "tom cruise laugh",
        "too damn high",
        "triggered feminist",
        "trojan horse",
        "trump interview",
        "trumpet boy",
        "trust nobody not even yourself",
        "trying to calculate how much sleep i can get",
        "turn up the volume",
        "twisted tea",
        "two buttons",
        "types of headaches meme",
        "undertaker",
        "unhelpful high school teacher",
        "uno draw 25 cards",
        "unsettled tom",
        "vince mcmahon",
        "virgin vs chad",
        "visible confusion",
        "wait thats illegal",
        "waiting skeleton",
        "we dont do that here",
        "we ride at dawn bitches",
        "weak vs strong spongebob",
        "well boys we did it blank is no more",
        "well yes but actually no",
        "what gives people feelings of power",
        "what if i told you",
        "what if you wanted to go to heaven",
        "whatcha got there",
        "whats my purpose - butter robot",
        "when x just right",
        "where banana",
        "who killed hannibal",
        "who wants to be a millionaire",
        "why are you gay",
        "why cant you just be normal",
        "willy wonka blank",
        "wolf of wallstreet",
        "wolverine remember",
        "x all the y",
        "x x everywhere",
        "yall got any more of",
        "yeah this is big brain time",
        "yo dawg heard you",
        "you cant defeat me",
        "you guys always act like youre better than me",
        "you guys are getting paid",
        "you guys are getting paid template",
        "you have no power here",
        "you know im something of a scientist myself",
        "you wouldnt get it",
        "young thug and lil durk troubleshooting",
        "yugioh card draw",
    ]
)
MEMES_TO_USE: List[str] = list(SEEN_MEMES - ALT_NAMES - DONT_USE_TEMPLATES)

