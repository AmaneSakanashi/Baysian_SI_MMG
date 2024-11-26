import numpy as np

from ..core import WorldCore
from ..wind import Wind
from ..utils import latlon2local


berth_latlon = [139.7916076222295, 35.61671139088672]
offshore_latlon = [139.8283315745129, 35.60869017029386]
berth_point = [0.0, 0.0]
offshore_point = latlon2local(offshore_latlon, berth_latlon)

ISLAND1 = np.array(
    [
        latlon2local([139.7901634501031, 35.59547095729461], berth_latlon),
        latlon2local([139.7905687114993, 35.59240236779743], berth_latlon),
        latlon2local([139.8084537164465, 35.58385719087195], berth_latlon),
        latlon2local([139.8134041296033, 35.57872221923013], berth_latlon),
        latlon2local([139.8322408686126, 35.58892085161354], berth_latlon),
        latlon2local([139.8251718768946, 35.6091884033497], berth_latlon),
        latlon2local([139.8247034161743, 35.61004444406954], berth_latlon),
        latlon2local([139.8105910645289, 35.61319409205115], berth_latlon),
        latlon2local([139.7985199376006, 35.60699952194481], berth_latlon),
    ]
)
ISLAND2 = np.array(
    [
        latlon2local([139.7840116371565, 35.58602314425631], berth_latlon),
        latlon2local([139.7786398782573, 35.58656934564117], berth_latlon),
        latlon2local([139.776798203173, 35.59028285963381], berth_latlon),
        latlon2local([139.7728459435342, 35.59550822606797], berth_latlon),
        latlon2local([139.7732470941035, 35.59565611337559], berth_latlon),
        latlon2local([139.763471106063, 35.60874108591264], berth_latlon),
        latlon2local([139.7639359010002, 35.60892386580851], berth_latlon),
        latlon2local([139.7620716607107, 35.61458115816069], berth_latlon),
        latlon2local([139.760188453251, 35.61587130188788], berth_latlon),
        latlon2local([139.7582130652306, 35.61767460503712], berth_latlon),
        latlon2local([139.753728980023, 35.61781667350453], berth_latlon),
        latlon2local([139.7556049817003, 35.58503075356742], berth_latlon),
        latlon2local([139.7581459805684, 35.57711652229445], berth_latlon),
        latlon2local([139.7833679404949, 35.57699027515476], berth_latlon),
        latlon2local([139.7852690085665, 35.57728259109539], berth_latlon),
        latlon2local([139.7872876663072, 35.57901184761122], berth_latlon),
        latlon2local([139.7874485141671, 35.58007671550587], berth_latlon),
        latlon2local([139.7870787788473, 35.58051048970084], berth_latlon),
    ]
)
ISLAND3 = np.array(
    [
        latlon2local([139.7809659397222, 35.62385809313109], berth_latlon),
        latlon2local([139.7846527723143, 35.62584606717458], berth_latlon),
        latlon2local([139.7849512478385, 35.62636713806176], berth_latlon),
        latlon2local([139.7770276850012, 35.63610721656408], berth_latlon),
        latlon2local([139.7765436909733, 35.63624875044586], berth_latlon),
        latlon2local([139.7731133877699, 35.63441106173274], berth_latlon),
        latlon2local([139.7725786633085, 35.63452735831193], berth_latlon),
        latlon2local([139.7723477843437, 35.63431167418384], berth_latlon),
        latlon2local([139.7717087322279, 35.63448662720862], berth_latlon),
        latlon2local([139.7713773224112, 35.63434308396597], berth_latlon),
        latlon2local([139.7659764469195, 35.62749633478186], berth_latlon),
        latlon2local([139.7663197182134, 35.62547620791409], berth_latlon),
        latlon2local([139.767567509762, 35.62581778044898], berth_latlon),
        latlon2local([139.7676396679746, 35.62206767403751], berth_latlon),
        latlon2local([139.770508084941, 35.61880735285107], berth_latlon),
        latlon2local([139.7698468301595, 35.61830678102704], berth_latlon),
        latlon2local([139.7743934201565, 35.61276256720598], berth_latlon),
        latlon2local([139.7767813472363, 35.61405876490618], berth_latlon),
        latlon2local([139.7867363385852, 35.60182971213103], berth_latlon),
        latlon2local([139.7869149917893, 35.60191182178781], berth_latlon),
        latlon2local([139.7926642084117, 35.60957154416009], berth_latlon),
    ]
)
ISLAND4 = np.array(
    [
        latlon2local([139.7944388561118, 35.61389566829962], berth_latlon),
        latlon2local([139.7946474444937, 35.613829864025], berth_latlon),
        latlon2local([139.794763027518, 35.61401779480705], berth_latlon),
        latlon2local([139.7946950258525, 35.61451015941409], berth_latlon),
        latlon2local([139.7950513588569, 35.61452052854359], berth_latlon),
        latlon2local([139.7952122295501, 35.61445017428096], berth_latlon),
        latlon2local([139.7957166985955, 35.61537922988396], berth_latlon),
        latlon2local([139.7960444049827, 35.61526991246244], berth_latlon),
        latlon2local([139.797016019837, 35.61718025669978], berth_latlon),
        latlon2local([139.797424703931, 35.61704481440141], berth_latlon),
        latlon2local([139.7976779400188, 35.61738554848476], berth_latlon),
        latlon2local([139.7979415164677, 35.61771305228306], berth_latlon),
        latlon2local([139.7979373565803, 35.61774266067761], berth_latlon),
        latlon2local([139.7965472409096, 35.61899023672212], berth_latlon),
        latlon2local([139.7967537860316, 35.61914758891601], berth_latlon),
        latlon2local([139.7967561940843, 35.61918519826108], berth_latlon),
        latlon2local([139.7955285151168, 35.62024394257446], berth_latlon),
        latlon2local([139.795735115838, 35.62039781548311], berth_latlon),
        latlon2local([139.7957384668882, 35.62043852839398], berth_latlon),
        latlon2local([139.789885322463, 35.62765903382191], berth_latlon),
        latlon2local([139.7852744980869, 35.6251399896173], berth_latlon),
    ]
)
ISLAND5 = np.array(
    [
        latlon2local([139.8274428433318, 35.61981841782596], berth_latlon),
        latlon2local([139.8306526329218, 35.61851282401651], berth_latlon),
        latlon2local([139.8280337649666, 35.61205783228903], berth_latlon),
        latlon2local([139.8284068027544, 35.61148278441541], berth_latlon),
        latlon2local([139.8355323526705, 35.61243055747283], berth_latlon),
        latlon2local([139.838611086022, 35.61420839631295], berth_latlon),
        latlon2local([139.840190284418, 35.63212389000152], berth_latlon),
        latlon2local([139.8257655868246, 35.6307241325717], berth_latlon),
    ]
)
ISLAND6 = np.array(
    [
        latlon2local([139.8176872525074, 35.61754040137832], berth_latlon),
        latlon2local([139.8220140862472, 35.61924232778082], berth_latlon),
        latlon2local([139.8230779398323, 35.62309860050571], berth_latlon),
        latlon2local([139.8205016529804, 35.6390319175060], berth_latlon),
        latlon2local([139.8183605227454, 35.6397417266667], berth_latlon),
        latlon2local([139.8099182664366, 35.6387196683003], berth_latlon),
        latlon2local([139.8092948763015, 35.63833908525579], berth_latlon),
        latlon2local([139.8072755754833, 35.62709718731983], berth_latlon),
        latlon2local([139.8095332132342, 35.6210414944472], berth_latlon),
    ]
)
ISLAND7 = np.array(
    [
        latlon2local([139.7986163533176, 35.62175352889899], berth_latlon),
        latlon2local([139.7989363343402, 35.6217534461743], berth_latlon),
        latlon2local([139.8022114947937, 35.62624034001264], berth_latlon),
        latlon2local([139.8004276317492, 35.6310375959602], berth_latlon),
        latlon2local([139.801457565064, 35.6328641316702], berth_latlon),
        latlon2local([139.8014854191158, 35.6364400961430], berth_latlon),
        latlon2local([139.7945111210763, 35.64488522408666], berth_latlon),
        latlon2local([139.7812615888726, 35.63879608664859], berth_latlon),
        latlon2local([139.7786050998327, 35.63736025562959], berth_latlon),
        latlon2local([139.7849452511322, 35.6295624913120], berth_latlon),
        latlon2local([139.7873630375335, 35.62727834019718], berth_latlon),
        latlon2local([139.7920051182255, 35.6298822685433], berth_latlon),
    ]
)


class Ariake(WorldCore):
    def __init__(self, wind=Wind()):
        self.wind = wind
        # state
        self.STATE_NAME = [
            "true_wind_speed [m/s]",
            "true_wind_direction [rad]",
        ]
        self.STATE_DIM = len(self.STATE_NAME)
        self.STATE_UPPER_BOUND = [np.inf, 2 * np.pi]
        self.STATE_LOWER_BOUND = [0.0, 0.0]
        # observation
        self.OBSERVATION_NAME = [
            "true_wind_speed_hat [m/s]",
            "true_wind_direction_hat [rad]",
        ]
        self.OBSERVATION_DIM = len(self.OBSERVATION_NAME)
        self.OBSERVATION_UPPER_BOUND = self.STATE_UPPER_BOUND
        self.OBSERVATION_LOWER_BOUND = self.STATE_LOWER_BOUND
        self.OBSERVATION_SCALE = [0.0, 0.0]
        # obstacles
        self.obstacle_polygons = [
            ISLAND1,
            ISLAND2,
            ISLAND3,
            ISLAND4,
            ISLAND5,
            ISLAND6,
            ISLAND7,
        ]

    def reset(self, state):
        w = state
        self.wind.reset(w)
        return state

    def step(self, dt, np_random=None):
        if np_random is None:
            np_random = np.random
        w_n = self.wind.step(dt, np_random=np_random)
        state_n = w_n
        return state_n

    def get_state(self):
        w = self.wind.get_state()
        state = w
        return state

    def observe_state(self, state, np_random=None):
        if np_random is None:
            np_random = np.random
        #
        additive_noise = np_random.normal(
            loc=np.zeros_like(self.OBSERVATION_SCALE),
            scale=self.OBSERVATION_SCALE,
        )
        observation = state + additive_noise
        return observation
