import numpy as np
import pytest

import deltares_wave_toolbox.cores.core_wavefunctions as core_wavefunctions


@pytest.mark.parametrize(
    ("f", "S", "Tps_exact"),
    (
        (0.5, 1, 2.0),
        ([1, 2], [1, 1], 1.0 / 1.5),
        ([1, 2], [1, 2], 0.5),
        ([1, 2], [2, 1], 1.0),
        ([0.5, 0.55, 0.61], [1, 0.5, 0.7], 2.0),
        ([0.35, 0.45, 0.5], [0.7, 0.5, 1], 2.0),
        ([0.35, 0.4, 0.6], [2, 1, 2], 1.0 / 0.35),
        ([0.35, 0.45, 0.5], [1, 1, 1], 1.0 / 0.45),
    ),
)
def test_compute_tps(f, S, Tps_exact):
    Tps_num = core_wavefunctions.compute_tps(f, S)
    assert Tps_num == Tps_exact


def test_create_spectrum_piersonmoskowitz():
    df = 0.005
    f = np.arange(0, 0.5 + df, df)
    Tp = 5
    fp = 1 / Tp
    hm0 = 1.5

    sVarDensCorrect = np.asarray(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            9.55679530937322e-209,
            3.94428212984427e-136,
            2.76673039591098e-92,
            1.42410683429054e-64,
            2.19655966924839e-46,
            4.60710983673921e-34,
            1.72999680559020e-25,
            2.20707462944724e-19,
            6.00511707957065e-15,
            1.13546125923339e-11,
            3.25417069343406e-09,
            2.39421178727747e-07,
            6.50433358025672e-06,
            8.42382395661323e-05,
            0.000624277107518723,
            0.00302237874347627,
            0.0105380860826621,
            0.0284553571068645,
            0.0628547293710449,
            0.118414172530380,
            0.196473436921618,
            0.294350393588782,
            0.406034986445398,
            0.523709812709215,
            0.639388656496310,
            0.746178686408509,
            0.838982579441132,
            0.914687973136063,
            0.971999039873340,
            1.01107781793020,
            1.03312752148557,
            1.04000187426388,
            1.03388251355566,
            1.01703710482862,
            0.991653571676174,
            0.959737755781218,
            0.923059568016337,
            0.883133629943840,
            0.841222746321815,
            0.798355232826517,
            0.755349603104197,
            0.712842171623788,
            0.671314709813217,
            0.631120446441252,
            0.592507506153065,
            0.555639413743401,
            0.520612628352854,
            0.487471269850509,
            0.456219303983910,
            0.426830496208023,
            0.399256449607969,
            0.373433025858275,
            0.349285420223526,
            0.326732128925118,
            0.305688013889911,
            0.286066638320711,
            0.267782017904021,
            0.250749907294008,
            0.234888719843147,
            0.220120160201611,
            0.206369634065203,
            0.193566486651166,
            0.181644111053601,
            0.170539959127208,
            0.160195480655095,
            0.150556010997394,
            0.141570622955125,
            0.133191955017354,
            0.125376025321253,
            0.118082038405050,
            0.111272190058547,
            0.104911474181284,
            0.0989674944684143,
            0.0934102828972167,
            0.0882121263331107,
            0.0833474020729817,
            0.0787924227634101,
            0.0745252908462915,
            0.0705257624737768,
            0.0667751206820879,
            0.0632560575065635,
            0.0599525646479420,
            0.0568498322542530,
            0.0539341553573163,
            0.0511928474927049,
            0.0486141610331378,
            0.0461872137745571,
            0.0439019213292061,
            0.0417489348990160,
            0.0397195840241100,
            0.0378058239241609,
            0.0360001870738999,
        ]
    )

    sVarDens = core_wavefunctions.create_spectrum_piersonmoskowitz(f, fp, hm0)

    assert sVarDens == pytest.approx(sVarDensCorrect, abs=1e-4)


def test_create_spectrum_jonswap():
    df = 0.005
    f = np.arange(0, 0.5 + df, df)
    Tp = 5
    fp = 1 / Tp
    hm0 = 1.5
    gammaPeak = 3.3

    sVarDensCorrect = [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        6.19757615890654e-209,
        2.55786465029219e-136,
        1.79422304075115e-92,
        9.23532447668775e-65,
        1.42446765856726e-46,
        2.98770802987004e-34,
        1.12190191483901e-25,
        1.43128660409561e-19,
        3.89431490776907e-15,
        7.36345961358535e-12,
        2.11032778811045e-09,
        1.55264494132236e-07,
        4.21805651648354e-06,
        5.46284490736276e-05,
        0.000404843343252061,
        0.00196001103434767,
        0.00683394743393034,
        0.0184533645171358,
        0.0407622699621248,
        0.0768009381775560,
        0.127480661241832,
        0.191273722026555,
        0.265114163520729,
        0.346539508111303,
        0.436974835680244,
        0.545695855960701,
        0.693326606193689,
        0.912178348928253,
        1.23493363476655,
        1.65370552953066,
        2.05372071616084,
        2.22565400024348,
        2.11480970297289,
        1.83489501329821,
        1.49512172177055,
        1.18509312852949,
        0.943587097067237,
        0.771309370154286,
        0.653310433427740,
        0.572798815876458,
        0.516225562252369,
        0.474076713867710,
        0.440254811626030,
        0.411174688237234,
        0.384917375017290,
        0.360555601587449,
        0.337685458682040,
        0.316144220305923,
        0.295863030754309,
        0.276800558291236,
        0.258917850036044,
        0.242171155893466,
        0.226511401413945,
        0.211885596145076,
        0.198238498232431,
        0.185514047397135,
        0.173656481727192,
        0.162611167969447,
        0.152325197199656,
        0.142747794924759,
        0.133830586781571,
        0.125527753184791,
        0.117796099594959,
        0.110595063576495,
        0.103886675348047,
        0.0976354849226873,
        0.0918084660417166,
        0.0863749047930424,
        0.0813062789643887,
        0.0765761327226981,
        0.0721599500598067,
        0.0680350295400757,
        0.0641803621787926,
        0.0605765137307805,
        0.0572055122445009,
        0.0540507414119960,
        0.0510968399984515,
        0.0483296074502661,
        0.0457359156439703,
        0.0433036266395201,
        0.0410215162319732,
        0.0388792030486376,
        0.0368670829091873,
        0.0349762681497870,
        0.0331985316056889,
        0.0315262549474839,
        0.0299523810722162,
        0.0284703702603339,
        0.0270741598217669,
        0.0257581269683649,
        0.0245170546647990,
        0.0233461002253077,
    ]

    sVarDens = core_wavefunctions.create_spectrum_jonswap(f, fp, hm0, gammaPeak)

    assert sVarDens == pytest.approx(sVarDensCorrect, abs=1e-4)


