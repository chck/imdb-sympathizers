namespace :tasks do
  desc 'Train all models'
  task :train, [:epochs] do |_, args|
    epochs = args[:epochs] || 4
    models = ['lstm_simple', 'lstm_simple_dropout', 'bilstm_dropout',
              'cnn_maxpool', 'cnn_maxpool_multifilter',
              'cnn_bilstm_dropout', 'cnn_bilstm_attention_dropout']
    optimizers = ['adam', 'nadam', 'rmsprop']
    models.product(optimizers) { |model, optimizer|
      sh "make train-repl MODEL=#{model} OPT=#{optimizer}"
      sleep 1
    }
  end
end
