from pathlib import Path
from typing import Dict, List, cast

import arrow
from pathlib2 import Path

LOAD_VERSION = "0.3.0"
TRAINING_VERSION = "0.3.0"

MONTH_TD = 60 * 60 * 24 * 30
WEEK_TD = 60 * 60 * 24 * 7
DAY_TD = 60 * 60 * 24
HOUR_TD = 60 * 60

IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNEL = 3
BATCH_SIZE = 32


def get_beginning() -> int:
    return cast(
        int,
        arrow.utcnow().shift(days=-31).replace(hour=0, minute=0, second=0).timestamp,
    )


FULL_SUB_LIST = ["dankmemes", "memes"]

PUSHSHIFT_URI = r"https://api.pushshift.io/reddit/search/submission?subreddit={}&after={}&before={}&size={}"

MODELS_REPO = "src/models/"
NOT_MEME_REPO = "src/data/not_a_meme/"
NOT_TEMPLATE_REPO = "src/data/not_a_template/"
MEMES_REPO = "src/data/memes/"
ALL_BLANKS_REPO = "src/data/all_blanks/"
BLANKS_REPO = "src/data/blanks/"
INCORRECT_REPO = "src/data/incorrect/"
Path(MODELS_REPO).mkdir(parents=True, exist_ok=True)  # type: ignore
Path(NOT_MEME_REPO).mkdir(parents=True, exist_ok=True)  # type: ignore
Path(NOT_TEMPLATE_REPO).mkdir(parents=True, exist_ok=True)  # type: ignore
Path(MEMES_REPO).mkdir(parents=True, exist_ok=True)  # type: ignore
Path(ALL_BLANKS_REPO).mkdir(parents=True, exist_ok=True)  # type: ignore
Path(BLANKS_REPO).mkdir(parents=True, exist_ok=True)  # type: ignore
Path(INCORRECT_REPO).mkdir(parents=True, exist_ok=True)  # type: ignore

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

STILL_NEEDED = ["Sad Linus"]
CAUSING_ISSUES = ["Talk To Spongebob"]

DONT_USE_TEMPLATES = set(
    [
        "hardworking guy",
        "blank pokemon card",
        "valentines day card meme",
        "sonic says",
        "facepalm bear",
        "i know that feel bro",
        "laughing villains",
        "among us not the imposter",
        "blank colored background",
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
        "lame pun coon",
        "silent movie card",
        "blank white template",
        "warning sign",
        "blank blue background",
        "black background",
        "keep calm and carry on black",
        "scumbag minecraft",
        "bart simpson - chalkboard",
        "impossibru guy original",
        "fat asian kid",
        "blank starter pack",
        "blank black",
        "keep calm and carry on aqua",
        "bart simpson peeking",
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
    ["computer guy facepalm", "computer guy", "computer horse"],
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
        "spongebob money",
        "i sleep real shit",
        "marvel civil war",
        "uno draw 25 cards",
        "yugioh card draw",
        "crying cat",
        "we ride at dawn bitches",
        "braveheart",
        "steve harvey laughing serious",
        "perfection",
        "smiling cat",
        "thats a paddlin",
        "im about to end this mans whole career",
        "success kid",
        "moe throws barney",
        "scared cat",
        "jojos walk",
        "finding neverland",
        "spongebob rainbow",
        "boardroom meeting suggestion",
        "mario bros views",
        "mr t pity the fool",
        "ptsd clarinet boy",
        "elmo fire",
        "peter griffin news",
        "bernie sanders once again asking",
        "straining kid",
        "who wants to be a millionaire",
        "is this a pigeon",
        "eye of sauron",
        "i want you",
        "bazooka squirrel",
        "twisted tea",
        "simba shadowy place",
        "grus plan",
        "a train hitting a school bus",
        "good fellas hilarious",
        "pokemon appears",
        "all my homies",
        "skinner out of touch",
        "young thug and lil durk troubleshooting",
        "bad luck brian",
        "car salesman slaps roof of car",
        "where banana",
        "wants to know your location",
        "am i a joke to you",
        "scooby doo mask reveal",
        "you underestimate my power",
        "farquaad pointing",
        "stop it patrick! youre scaring him!",
        "doge",
        "wolf of wallstreet",
        "get in loser",
        "alright gentlemen we need a new idea",
        "rick and morty-extra steps",
        "sue sylvester",
        "the rock driving",
        "virgin vs chad",
        "npc",
        "back in my day",
        "first world problems",
        "evil kermit",
        "my time has come",
        "skinner pathetic",
        "and i took that personally",
        "angry turkish man playing cards meme",
        "this little manuever is gonna cost us 51 years",
        "kill yourself guy",
        "x x everywhere",
        "dj khaled suffering from success meme",
        "put it somewhere else patrick",
        "bro not cool",
        "slenderman",
        "malicious advice mallard",
        "its been 84 years",
        "spiderman glasses",
        "and everybody loses their minds",
        "blackjack and hookers",
        "baby yoda die trash",
        "jordan peele sweating",
        "sully wazowski",
        "this is fine",
        "extra-hell",
        "look at me",
        "yall got any more of",
        "batman slapping robin",
        "high five drown",
        "ill have you know spongebob",
        "if those kids could read theyd be very upset",
        "the mandalorian",
        "do you want ants archer",
        "you have no power here",
        "change my mind",
        "kim jong un sad",
        "i am inevitable",
        "mike wazowski trying to explain",
        "billys fbi agent",
        "every day i wake up",
        "angry pakistani fan",
        "lumbergh",
        "star wars yoda",
        "black guy disappearing",
        "you know im something of a scientist myself",
        "futurama fry",
        "grumpy cat",
        "ah shit here we go again",
        "flex tape",
        "the most interesting cat in the world",
        "guy holding cardboard sign",
        "domino effect",
        "homer simpsons back fat",
        "meg family guy better than me",
        "thanos perfectly balanced",
        "ermahgerd berks",
        "borat",
        "thanos infinity stones",
        "bro explaining",
        "trust nobody not even yourself",
        "waiting skeleton",
        "original stoner dog",
        "third world skeptical kid",
        "fist pump baby",
        "guy pouring olive oil on the salad",
        "spongebob ight imma head out",
        "crying peter parker",
        "samuel l jackson",
        "inhaling seagull",
        "they hated jesus meme",
        "take a seat cat",
        "yeah this is big brain time",
        "advice yoda",
        "arthur fist",
        "madea with gun",
        "stonks helth",
        "you wouldnt get it",
        "boy and girl texting",
        "obi wan kenobi",
        "aint nobody got time for that",
        "this is worthless",
        "why are you gay",
        "persian cat room guardian",
        "gotta go cat",
        "ppap",
        "well boys we did it blank is no more",
        "nuclear explosion",
        "kombucha girl",
        "no i dont think i will",
        "trump interview",
        "homer disappears into bush",
        "the floor is",
        "running away balloon",
        "mel gibson and jesus christ",
        "my heart blank",
        "leonardo dicaprio cheers",
        "calculating meme",
        "leonardo dicaprio pointing",
        "blank nut button",
        "scumbag brain",
        "jack sparrow being chased",
        "scumbag steve",
        "business cat",
        "i bet hes thinking about other women",
        "squidward",
        "advice dog",
        "say the line bart! simpsons",
        "sleepy donald duck in bed",
        "crying michael jordan",
        "mr bean",
        "communist bugs bunny",
        "wait its all",
        "angela scared dwight",
        "the what",
        "disappointed black guy",
        "metal jesus",
        "pimples zero!",
        "satisfied seal",
        "confused gandalf",
        "impostor of the vent",
        "bad pun dog",
        "mexican word of the day",
        "mother ignoring kid drowning in a pool",
        "angry baby",
        "honey its time to x",
        "soldier jump spetznaz",
        "imagination spongebob",
        "money money",
        "whatcha got there",
        "pentagon hexagon octagon",
        "why cant you just be normal",
        "captain america so you",
        "you made thisi made this",
        "sparta leonidas",
        "gang bang",
        "weak vs strong spongebob",
        "persian cat room guardian single",
        "aaaaand its gone",
        "oh dear dear gorgeus",
        "cross eyed spongebob",
        "patrick not my wallet",
        "distracted boyfriend",
        "spongebob waiting",
        "idiot skull",
        "oh boy here i go killing again",
        "squidward chair",
        "but thats none of my business",
        "spiderman peter parker",
        "left exit 12 off ramp",
        "mr incredible mad",
        "bronze medal",
        "triggered feminist",
        "joe exotic financially recover",
        "when x just right",
        "internet explorer",
        "bird box",
        "dog vs werewolf",
        "star wars no",
        "trojan horse",
        "buff doge vs crying cheems",
        "thanos impossible",
        "turn up the volume",
        "spongebob diapers meme",
        "money man",
        "here it come meme",
        "dog on fire",
        "spongebob strong",
        "all right then keep your secrets",
        "mr bean copying",
        "you cant defeat me",
        "why is the fbi here",
        "admiral ackbar",
        "unhelpful high school teacher",
        "captain america elevator",
        "one does not simply",
        "team rocket",
        "what if you wanted to go to heaven",
        "panik kalm panik",
        "dad joke dog",
        "disaster girl",
        "what if i told you",
        "pink guy vs bane",
        "srgrafo dude wtf",
        "the silent protector",
        "chocolate spongebob",
        "highdrunk guy",
        "hey you going to sleep",
        "short satisfaction vs truth",
        "you know the rules its time to die",
        "relief",
        "skeleton waiting",
        "say what again",
        "grandma finds the internet",
        "adios",
        "let me in",
        "snape",
        "the office bankruptcy",
        "epic handshake",
        "angry chef gordon ramsay",
        "kermit window",
        "whats my purpose - butter robot",
        "mocking spongebob",
        "oprah you get a",
        "you guys are getting paid template",
        "happy star congratulations",
        "first world problems cat",
        "shut up and take my money fry",
        "bike fall",
        "joaquin phoenix joker car",
        "the most interesting man in the world",
        "rick and carl",
        "we dont do that here",
        "ill just wait here",
        "spongebob hype stand",
        "the cooler daniel",
        "annoyed bird",
        "rainbow six - fuze the hostage",
        "philosoraptor",
        "the scroll of truth",
        "visible confusion",
        "gru gun",
        "guy with sand in the hands of despair",
        "lady screams at cat",
        "charlie day",
        "fancy pooh",
        "maybe i am a monster",
        "phoebe joey",
        "for christmas i want a dragon",
        "listen here you little shit bird",
        "does your dog bite",
        "speechless colbert face",
        "i find your lack of faith disturbing",
        "brace yourself",
        "chuck norris",
        "gadsden flag",
        "dabbing dude",
        "zuckerberg",
        "eddie murphy thinking",
        "stonks",
        "elmo nuclear explosion",
        "unsettled tom",
        "unhappy baby",
        "surprised koala",
        "jack sparrow you have heard of me",
        "and just like that",
        "bernie sanders reaction nuked",
        "no i dont think i will",
        "trumpet boy",
        "god",
        "you guys are getting paid",
        "expanding brain",
        "pretending to be happy hiding crying behind a mask",
        "i am the senate",
        "laughing leo",
        "lisa simpson coffee that x shit",
        "what gives people feelings of power",
        "mr bean waiting",
        "surprised pikachu",
        "omg karen",
        "willy wonka blank",
        "anthony adams rubbing hands",
        "mr krabs blur meme",
        "im the dumbest man alive",
        "knights of the round table",
        "pepe the frog",
        "life is good but it can be better",
        "cnn breaking news template",
        "it aint much but its honest work",
        "crying baby yoda",
        "third world success kid",
        "blank kermit waiting",
        "confucius says",
        "the loudest sounds on earth",
        "red pill blue pill",
        "undertaker",
        "donald trump",
        "yeet the child",
        "tom cruise laugh",
        "bernie mittens",
        "feels good man",
        "joe biden",
        "consuela",
        "cmon do something",
        "oh no! anyway",
        "captain picard facepalm",
        "its finally over",
        "foul bachelor frog",
        "aw yeah rage face",
        "aww did someone get addicted to crack",
        "so you have chosen death",
        "challenge accepted rage face",
        "amateurs",
        "dont you squidward",
        "baby yoda",
        "rod serling twilight zone",
        "simpsons so far",
        "suprised patrick",
        "salt bae",
        "baby yoda tea",
        "see nobody cares",
        "black guy confused",
        "sure grandma lets get you to bed",
        "anime wall punch",
        "i believe in supremacy",
        "overly attached girlfriend",
        "how tough are you",
        "i do one push-up",
        "steve buscemi fellow kids",
        "first time",
        "anime girl hiding from terminator",
        "so anyway i started blasting",
        "x all the y",
        "fallout hold up",
        "spongebob burning paper",
        "clown applying makeup",
        "math is math",
        "ancient aliens",
        "friendship ended",
        "call an ambulance but not for me",
        "dr evil",
        "inception",
        "thanos what did it cost",
        "dinkleberg",
        "me and the boys",
        "man giving sword to larger man",
        "mario hammer smash",
        "its free real estate",
        "prisoners blank",
        "well yes but actually no",
        "blankie the shocked dog",
        "dwight schrute",
        "lisa simpsons presentation",
        "the boiler room of hell",
        "monkey looking away",
        "dont make me tap the sign",
        "squidward window",
        "the future world if",
        "chuckles im in danger",
        "1990s first world problems",
        "karate kyle",
        "not stonks",
        "yo dawg heard you",
        "jersey santa",
        "grant gustin over grave",
        "jason momoa henry cavill meme",
        "pointing mirror guy",
        "surprised baby yoda",
        "i dont want to play with you anymore",
        "nemo seagulls mine",
        "is for me",
        "computer horse",
        "patrick smart dumb",
        "buddy christ",
        "wtf",
        "sarcastically surprised kirk",
        "cool cat stroll",
        "who killed hannibal",
        "note passing",
        "big book small book",
        "shrek for five minutes",
        "two buttons",
        "girl running",
        "genie rules meme",
        "buddy the elf",
        "wolverine remember",
        "joey from friends",
        "spiderman laugh",
        "and thats a fact",
        "ew i stepped in shit",
        "you the real mvp 2",
        "ptsd chihuahua",
        "i wish all the x a very pleasant evening",
        "they dont know",
        "thomas had never seen such bullshit before",
        "wait thats illegal",
        "nobody is born cool",
        "metronome",
        "drake noyes",
        "arrogant rich man",
        "the office handshake",
        "open the gate a little",
        "assassination chain",
        "this is where id put my trophy if i had one",
        "elmo cocaine",
        "laughing men in suits",
        "nemo birds",
        "krusty krab vs chum bucket",
        "finally",
        "average blank fan vs average blank enjoyer",
        "black girl wat",
        "spiderman",
        "if you know what i mean bean",
        "dw sign wont stop me because i cant read",
        "floating boy chasing running boy",
        "always has been",
        "gollum",
        "second breakfast",
        "pepperidge farm remembers",
        "theyre the same picture",
        "cute cat",
        "minor mistake marvin",
        "trying to calculate how much sleep i can get",
        "sweating bullets",
        "sad baby yoda",
        "grim reaper knocking door",
        "types of headaches meme",
        "hide the pain harold",
        "i should buy a boat cat",
        "you guys always act like youre better than me",
        "la noire press x to doubt",
        "spongegar",
        "but it was me dio",
        "modern problems",
        "allow us to introduce ourselves",
        "i fear no man",
        "confused screaming",
        "confused unga bunga",
        "socially awkward awesome penguin",
        "finally! a worthy opponent!",
        "gordon ramsay some good food",
        "this morgan freeman",
        "vince mcmahon",
        "mugatu so hot right now",
        "excuse me wtf blank template",
        "jack nicholson the shining snow",
        "kevin hart",
        "afraid to ask andy",
        "me explaining to my mom",
        "soldier protecting sleeping child",
        "jim halpert explains",
        "this is brilliant but i like this",
        "too damn high",
        "gangnam style",
        "meme man",
        "thor is he though",
        "i am speed",
        "cereal guy",
        "scared kid",
        "daring today arent we squidward",
        "laughing wolf",
        "sad keanu",
    ]
)
MEMES_TO_USE: List[str] = list(
    COMPRESSED_NAMES.union(SEEN_MEMES - ALT_NAMES) - DONT_USE_TEMPLATES
)
