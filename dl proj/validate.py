import torch
import argparse
import pickle 
from data_loader import get_loader 
from torchvision import transforms 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn as nn
import time
import pdb
import math
from pathlib import Path
home = str(Path.home())

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def main(args):
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
        
    # Build models
    encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    
    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    
    # Build data loader
    data_loader = get_loader(args.image_dir, args.caption_path, vocab, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers)
    print('val_loader length = {}'.format(len(data_loader)))
    
    val_loss = 0
    start = time.time()
    for i, (images, captions, lengths) in enumerate(data_loader):
        images = images.to(device)
        captions = captions.to(device)
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
        # Forward
        features = encoder(images)
        outputs = decoder(features, captions, lengths)
        val_loss += criterion(outputs, targets)
        
        # Print log info
        if i % args.log_step == 0:
            print('step {}/{}, time {}'.format(i, len(data_loader), timeSince(start)))    
    
    val_loss = val_loss/len(data_loader)
    print('val_loss = {:.3f}'.format(val_loss))
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_path', type=str, default= home+ '/data2/models/encoder-5-3000.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default= home+ '/data2/models/decoder-5-3000.ckpt', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default= home+ '/data2/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default= home+ '/data2/val_resized2014', help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default= home+ '/data2/annotations/captions_val2014.json', help='path for train annotation json file')
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    
    args = parser.parse_args()
    print(args)
    main(args)
    
    
    
    
    
    
    
    
    
   
    