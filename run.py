from torch.utils.data import DataLoader
from dataset_loaders import *
import evaluation
from emvos import EMVOS
from trainer import Trainer
from optparse import OptionParser
import warnings
warnings.filterwarnings('ignore')


parser = OptionParser()
parser.add_option('--train', action='store_true', dest='train', default=None)
parser.add_option('--test', action='store_true', dest='test', default=None)
(options, args) = parser.parse_args()

torch.cuda.set_device(0)


##################
# Train
##################
def train_coco(model):
    train_set = TrainCOCO('../DB/COCO', output_size=(384, 384), clip_l=5, clip_n=128)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=16, num_workers=4)
    val_set = TestDAVIS('../DB/DAVIS', '2017', 'val')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    trainer = Trainer(model, optimizer, train_loader, val_set, save_name='coco', save_step=4000, val_step=400)
    # trainer.load_checkpoint('coco', 8000, gpu=0)
    trainer.train(8000)


def train_davis(model):
    train_set = TrainDAVIS('../DB/DAVIS', '2017', 'train', output_size=(384, 384), clip_l=10, clip_n=128)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=8, num_workers=4)
    val_set = TestDAVIS('../DB/DAVIS', '2017', 'val')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    trainer = Trainer(model, optimizer, train_loader, val_set, save_name='coco_davis', save_step=2000, val_step=200)
    trainer.train(4000)


def train_ytvos(model):
    train_set = TrainYTVOS('../DB/YTVOS18', output_size=(384, 384), clip_l=10, clip_n=128)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=8, num_workers=4)
    val_set = None
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    trainer = Trainer(model, optimizer, train_loader, val_set, save_name='coco_ytvos', save_step=2000, val_step=200)
    trainer.train(8000)


##################
# Test
##################
def test(model):
    datasets = {
        'DAVIS16_val': TestDAVIS('../DB/DAVIS', '2016', 'val'),
        'DAVIS17_val': TestDAVIS('../DB/DAVIS', '2017', 'val'),
        'DAVIS17_test-dev': TestDAVIS('../DB/DAVIS', '2017', 'test-dev'),
        # 'YTVOS18_val': TestYTVOS('../DB/YTVOS18')
    }

    for key, dataset in datasets.items():
        evaluator = evaluation.Evaluator(dataset)
        evaluator.evaluate(model, os.path.join('outputs', key))


def main():
    #########################
    # for reproducibility
    #########################
    seed = 19971007
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # define model
    model = EMVOS().eval()

    # training stage
    if options.train:
        train_coco(model)
        train_davis(model)
        # train_ytvos(model)

    # testing stage
    if options.test:
        model.load_state_dict(torch.load('trained_model/coco_davis_best.pth', map_location='cuda:0'))
        with torch.no_grad():
            test(model)


if __name__ == '__main__':
    main()
